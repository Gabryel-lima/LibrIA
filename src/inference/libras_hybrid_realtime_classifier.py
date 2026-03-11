"""Classificador híbrido com arbitragem entre modelos estático e temporal."""

import os
import pickle
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2 as cv
import numpy as np

from config.settings import CAMERA_CONFIG, FEATURE_DIMENSION, FEATURE_MODE, HYBRID_INFERENCE_CONFIG, INFERENCE_CONFIG, LSTM_CONFIG
from utils.helpers import extract_landmarks_by_mode, load_camera_calibration, preprocess_frame

from .prediction_merger import PredictionEvent, PredictionMerger

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_ERROR = e

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    TENSORFLOW_AVAILABLE = False
    TENSORFLOW_ERROR = e


class LibrasHybridRealtimeClassifier:
    """Executa um único loop de câmera com arbitragem entre RF e LSTM."""

    def __init__(
        self,
        static_model_path: str = './model/model.pickle',
        temporal_model_path: str = LSTM_CONFIG['model_path'],
        temporal_label_map_path: str = LSTM_CONFIG['label_map_path'],
        prediction_interval: int = INFERENCE_CONFIG['prediction_interval'],
    ):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                'MediaPipe não disponível para inferência híbrida. '
                f"Motivo: {type(MEDIAPIPE_ERROR).__name__}: {MEDIAPIPE_ERROR}"
            )
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError(
                'TensorFlow não disponível para inferência híbrida. '
                f"Motivo: {type(TENSORFLOW_ERROR).__name__}: {TENSORFLOW_ERROR}"
            )

        self.static_model_path = static_model_path
        self.temporal_model_path = temporal_model_path
        self.temporal_label_map_path = temporal_label_map_path
        self.prediction_interval = prediction_interval
        self.feature_mode = FEATURE_MODE
        self.feature_dimension = FEATURE_DIMENSION
        self.sequence_length = LSTM_CONFIG['sequence_length']
        self.allowed_temporal_classes = {label.upper() for label in LSTM_CONFIG.get('allowed_classes', [])}

        self.static_model, self.static_metadata = self._load_static_model()
        self.temporal_model = self._load_temporal_model()
        self.temporal_label_map, self.temporal_metadata = self._load_temporal_metadata()
        self._validate_temporal_metadata()

        self.camera_calibration = self._load_camera_calibration()
        self.sequence_buffer: Deque[np.ndarray] = deque(maxlen=self.sequence_length)
        self.sequence_timestamps: Deque[float] = deque(maxlen=self.sequence_length)
        self.frame_index = 0
        self.alphabet_dict = {i: chr(65 + i) for i in range(26)}
        self.last_static_candidate: Optional[PredictionEvent] = None
        self.last_temporal_candidate: Optional[PredictionEvent] = None
        self.last_output: Optional[PredictionEvent] = None

        self.merger = PredictionMerger(
            temporal_priority_classes=HYBRID_INFERENCE_CONFIG['temporal_priority_classes'],
            temporal_confidence_threshold=HYBRID_INFERENCE_CONFIG['temporal_confidence_threshold'],
            static_confidence_threshold=HYBRID_INFERENCE_CONFIG['static_confidence_threshold'],
            cooldown_seconds=HYBRID_INFERENCE_CONFIG['prediction_cooldown_seconds'],
            history_size=HYBRID_INFERENCE_CONFIG['overlay_history_size'],
        )

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=INFERENCE_CONFIG['min_detection_confidence'],
            min_tracking_confidence=0.5,
        )

    def _load_static_model(self) -> Tuple[object, dict]:
        if not os.path.exists(self.static_model_path):
            raise FileNotFoundError(f'Modelo estático não encontrado: {self.static_model_path}')
        with open(self.static_model_path, 'rb') as file_obj:
            model_dict = pickle.load(file_obj)
        return model_dict.get('model'), {key: value for key, value in model_dict.items() if key != 'model'}

    def _load_temporal_model(self):
        if not os.path.exists(self.temporal_model_path):
            raise FileNotFoundError(f'Modelo temporal não encontrado: {self.temporal_model_path}')
        return tf.keras.models.load_model(self.temporal_model_path)

    def _load_temporal_metadata(self) -> Tuple[dict, dict]:
        if not os.path.exists(self.temporal_label_map_path):
            raise FileNotFoundError(f'Metadados temporais não encontrados: {self.temporal_label_map_path}')
        with open(self.temporal_label_map_path, 'rb') as file_obj:
            metadata = pickle.load(file_obj)
        return metadata.get('label_map', {}), metadata

    def _validate_temporal_metadata(self):
        trained_classes = {label.upper() for label in self.temporal_label_map.values()}
        if self.allowed_temporal_classes and not trained_classes.issubset(self.allowed_temporal_classes):
            raise ValueError(
                'Modelo temporal contém classes fora do conjunto permitido: '
                f"{sorted(trained_classes.difference(self.allowed_temporal_classes))}"
            )

        metadata_feature_size = self.temporal_metadata.get('feature_size')
        metadata_sequence_length = self.temporal_metadata.get('sequence_length')
        if metadata_feature_size and int(metadata_feature_size) != self.feature_dimension:
            raise ValueError(
                'Feature size incompatível entre híbrido e metadados do LSTM: '
                f'{metadata_feature_size} != {self.feature_dimension}'
            )
        if metadata_sequence_length and int(metadata_sequence_length) != self.sequence_length:
            raise ValueError(
                'Sequence length incompatível entre híbrido e metadados do LSTM: '
                f'{metadata_sequence_length} != {self.sequence_length}'
            )

    def _load_camera_calibration(self):
        if not CAMERA_CONFIG['enabled']:
            return None
        return load_camera_calibration(
            CAMERA_CONFIG['camera_matrix_path'],
            CAMERA_CONFIG['dist_coeffs_path'],
        )

    def _predict_static(self, features: np.ndarray, timestamp: float) -> Optional[PredictionEvent]:
        if self.frame_index % self.prediction_interval != 0:
            return None

        prediction = self.static_model.predict([features])[0]
        try:
            confidence = float(np.max(self.static_model.predict_proba([features])[0]))
        except AttributeError:
            confidence = 0.0

        token = self.alphabet_dict.get(int(prediction), '?')
        return PredictionEvent(
            token=token,
            confidence=confidence,
            source='static',
            start_time=timestamp,
            end_time=timestamp,
            frame_index=self.frame_index,
        )

    def _predict_temporal(self) -> Optional[PredictionEvent]:
        if len(self.sequence_buffer) < self.sequence_length:
            return None

        input_sequence = np.expand_dims(np.asarray(self.sequence_buffer, dtype=np.float32), axis=0)
        probabilities = self.temporal_model.predict(input_sequence, verbose=0)[0]
        label_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[label_idx])
        token = self.temporal_label_map.get(label_idx, str(label_idx))
        return PredictionEvent(
            token=token,
            confidence=confidence,
            source='temporal',
            start_time=float(self.sequence_timestamps[0]),
            end_time=float(self.sequence_timestamps[-1]),
            frame_index=self.frame_index,
        )

    def _append_temporal_features(self, features: np.ndarray, timestamp: float):
        self.sequence_buffer.append(features)
        self.sequence_timestamps.append(timestamp)

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        cv.putText(frame, 'LibrIA - Inferencia Hibrida', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(
            frame,
            f'Buffer temporal: {len(self.sequence_buffer)}/{self.sequence_length}',
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 220),
            2,
        )

        if self.last_output is not None:
            cv.putText(
                frame,
                f'Saida: {self.last_output.token} [{self.last_output.source}] {self.last_output.confidence:.2%}',
                (10, 95),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

        if self.last_static_candidate is not None:
            cv.putText(
                frame,
                f'Estatico: {self.last_static_candidate.token} {self.last_static_candidate.confidence:.2%}',
                (10, 125),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 255, 180),
                2,
            )

        if self.last_temporal_candidate is not None:
            cv.putText(
                frame,
                f'Temporal: {self.last_temporal_candidate.token} {self.last_temporal_candidate.confidence:.2%}',
                (10, 150),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 180, 255),
                2,
            )

        return frame

    def start_classification(self, camera_index: int = 0):
        cap = cv.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f'Não foi possível abrir a câmera {camera_index}')

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = preprocess_frame(frame, self.camera_calibration)
                timestamp = time.time()
                results = self.hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )
                    features = extract_landmarks_by_mode(hand_landmarks.landmark, self.feature_mode)
                else:
                    features = np.zeros(self.feature_dimension, dtype=np.float32)

                self._append_temporal_features(features, timestamp)

                static_event = self._predict_static(features, timestamp)
                if static_event is not None:
                    self.last_static_candidate = static_event
                    merged_event = self.merger.submit(static_event)
                    if merged_event is not None:
                        self.last_output = merged_event

                temporal_event = self._predict_temporal()
                if temporal_event is not None:
                    self.last_temporal_candidate = temporal_event
                    merged_event = self.merger.submit(temporal_event)
                    if merged_event is not None:
                        self.last_output = merged_event

                frame = self._draw_overlay(frame)
                cv.imshow('LibrIA - Inferencia Hibrida', frame)

                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                self.frame_index += 1
        finally:
            cap.release()
            cv.destroyAllWindows()
            self.hands.close()


def main():
    classifier = LibrasHybridRealtimeClassifier()
    classifier.start_classification()


if __name__ == '__main__':
    main()