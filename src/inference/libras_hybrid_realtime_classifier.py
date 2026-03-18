"""Classificador híbrido com arbitragem entre modelos estático e temporal."""

from warnings import filterwarnings

# Ignora especificamente o UserWarning do Protobuf
filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

import os
import pickle
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2 as cv
import numpy as np

from config.settings import CAMERA_CONFIG, FEATURE_MODE, HYBRID_INFERENCE_CONFIG, INFERENCE_CONFIG, LSTM_CONFIG
from utils.helpers import (
    enrich_model_metadata,
    extract_landmarks_by_mode,
    get_feature_dimension,
    infer_feature_mode_from_dimension,
    load_camera_calibration,
    patch_legacy_sklearn_model,
    preprocess_frame,
)

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
        self.sequence_length = LSTM_CONFIG['sequence_length']
        self.allowed_temporal_classes = {label.upper() for label in LSTM_CONFIG.get('allowed_classes', [])}

        self.static_model, self.static_metadata = self._load_static_model()
        self.temporal_model = self._load_temporal_model()
        self.temporal_label_map, self.temporal_metadata = self._load_temporal_metadata()
        self.static_feature_mode = self._resolve_static_feature_mode()
        self.static_feature_dimension = get_feature_dimension(self.static_feature_mode)
        self.temporal_feature_mode = self._resolve_temporal_feature_mode()
        self.temporal_feature_dimension = get_feature_dimension(self.temporal_feature_mode)
        self.feature_mode = self.temporal_feature_mode
        self.feature_dimension = self.temporal_feature_dimension
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
        model = patch_legacy_sklearn_model(model_dict.get('model'))
        metadata = enrich_model_metadata(model, {key: value for key, value in model_dict.items() if key != 'model'})
        return model, metadata

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

    def _resolve_static_feature_mode(self) -> str:
        metadata_mode = self.static_metadata.get('feature_mode')
        if metadata_mode:
            return metadata_mode

        metadata_num_features = self.static_metadata.get('num_features')
        if metadata_num_features is not None:
            return infer_feature_mode_from_dimension(int(metadata_num_features))

        if hasattr(self.static_model, 'n_features_in_'):
            return infer_feature_mode_from_dimension(int(self.static_model.n_features_in_))

        return FEATURE_MODE

    def _resolve_temporal_feature_mode(self) -> str:
        metadata_mode = self.temporal_metadata.get('feature_mode')
        if metadata_mode:
            return metadata_mode

        metadata_feature_size = self.temporal_metadata.get('feature_size')
        if metadata_feature_size is not None:
            return infer_feature_mode_from_dimension(int(metadata_feature_size))

        return FEATURE_MODE

    def _validate_temporal_metadata(self):
        trained_classes = {label.upper() for label in self.temporal_label_map.values()}
        if self.allowed_temporal_classes and not trained_classes.issubset(self.allowed_temporal_classes):
            raise ValueError(
                'Modelo temporal contém classes fora do conjunto permitido: '
                f"{sorted(trained_classes.difference(self.allowed_temporal_classes))}"
            )

        metadata_feature_size = self.temporal_metadata.get('feature_size')
        metadata_sequence_length = self.temporal_metadata.get('sequence_length')
        if metadata_feature_size and int(metadata_feature_size) != self.temporal_feature_dimension:
            raise ValueError(
                'Feature size incompatível entre híbrido e metadados do LSTM: '
                f'{metadata_feature_size} != {self.temporal_feature_dimension}'
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

        mirrored_features = self._mirror_features(features, self.static_feature_mode)
        mirrored_prediction = self.static_model.predict([mirrored_features])[0]
        try:
            mirrored_confidence = float(np.max(self.static_model.predict_proba([mirrored_features])[0]))
        except AttributeError:
            mirrored_confidence = 0.0

        if mirrored_confidence > confidence:
            prediction = mirrored_prediction
            confidence = mirrored_confidence

        token = self._static_prediction_to_token(prediction)
        return PredictionEvent(
            token=token,
            confidence=confidence,
            source='static',
            start_time=timestamp,
            end_time=timestamp,
            frame_index=self.frame_index,
        )

    def _static_prediction_to_token(self, prediction) -> str:
        """Normalize static model prediction to a single-letter token.

        Handles cases where the model returns an integer index or a string label
        (e.g., 'G'). Falls back to '?' when conversion fails.
        """
        # Numpy scalar or Python int
        try:
            if isinstance(prediction, (int, np.integer)):
                return self.alphabet_dict.get(int(prediction), '?')
        except Exception:
            pass

        # Bytes -> decode
        if isinstance(prediction, (bytes, bytearray)):
            try:
                prediction = prediction.decode('utf-8')
            except Exception:
                return '?'

        # String labels (e.g., 'G') or numeric strings
        if isinstance(prediction, str):
            p = prediction.strip()
            # If single alpha character, return uppercase
            if len(p) == 1 and p.isalpha():
                return p.upper()
            # Try numeric string -> index
            try:
                idx = int(p)
                return self.alphabet_dict.get(idx, '?')
            except Exception:
                # Unknown string format, return as-is (uppercased) if plausible
                return p.upper() if p else '?'

        # Fallback
        return '?'

    def _predict_temporal(self) -> Optional[PredictionEvent]:
        if len(self.sequence_buffer) < self.sequence_length:
            return None

        input_sequence = np.expand_dims(np.asarray(self.sequence_buffer, dtype=np.float32), axis=0)
        probabilities = self.temporal_model.predict(input_sequence, verbose=0)[0]
        label_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[label_idx])

        mirrored_sequence = self._mirror_sequence(np.asarray(self.sequence_buffer, dtype=np.float32))
        mirrored_probabilities = self.temporal_model.predict(
            np.expand_dims(mirrored_sequence, axis=0),
            verbose=0,
        )[0]
        mirrored_label_idx = int(np.argmax(mirrored_probabilities))
        mirrored_confidence = float(mirrored_probabilities[mirrored_label_idx])

        if mirrored_confidence > confidence:
            label_idx = mirrored_label_idx
            confidence = mirrored_confidence

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

    def _mirror_features(self, features: np.ndarray, feature_mode: Optional[str] = None) -> np.ndarray:
        mode = feature_mode or getattr(self, 'feature_mode', FEATURE_MODE)
        mirrored = np.asarray(features, dtype=np.float32).copy().reshape(-1)

        if mirrored.size % 3 != 0:
            return mirrored

        if mode == 'wrist_relative':
            mirrored[0::3] = -mirrored[0::3]
        else:
            mirrored[0::3] = 1.0 - mirrored[0::3]
        return mirrored

    def _mirror_sequence(self, sequence: np.ndarray) -> np.ndarray:
        sequence_array = np.asarray(sequence, dtype=np.float32)
        if sequence_array.ndim != 2:
            sequence_array = sequence_array.reshape(self.sequence_length, -1)

        mirrored = sequence_array.copy()
        if mirrored.shape[1] % 3 != 0:
            return mirrored

        if self.temporal_feature_mode == 'wrist_relative':
            mirrored[:, 0::3] = -mirrored[:, 0::3]
        else:
            mirrored[:, 0::3] = 1.0 - mirrored[:, 0::3]
        return mirrored

    def _extract_features(self, landmarks) -> Tuple[np.ndarray, np.ndarray]:
        static_features = extract_landmarks_by_mode(landmarks, self.static_feature_mode)

        if self.static_feature_mode == self.temporal_feature_mode:
            return static_features, static_features.copy()

        temporal_features = extract_landmarks_by_mode(landmarks, self.temporal_feature_mode)
        return static_features, temporal_features

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        cv.putText(frame, 'LibrIA - Inferencia Hibrida', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv.putText(
            frame,
            f'Buffer temporal: {len(self.sequence_buffer)}/{self.sequence_length}',
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0), # Quem foi o gênio que inverteu o padrão RGB?
            2,
        )

        # if self.last_output is not None:
        #     cv.putText(
        #         frame,
        #         f'Saida: {self.last_output.token} [{self.last_output.source}] {self.last_output.confidence:.2%}',
        #         (10, 95),
        #         cv.FONT_HERSHEY_SIMPLEX,
        #         0.7,
        #         (255, 0, 0), # Quem foi o gênio que inverteu o padrão RGB?
        #         2,
        #     )

        if self.last_temporal_candidate is not None:
            cv.putText(
                frame,
                f'Temporal: {self.last_temporal_candidate.token} {self.last_temporal_candidate.confidence:.2%}',
                (10, 95),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0), # Quem foi o gênio que inverteu o padrão RGB?
                2,
            )

        if self.last_static_candidate is not None:
            cv.putText(
                frame,
                f'Estatico: {self.last_static_candidate.token} {self.last_static_candidate.confidence:.2%}',
                (10, 125),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0), # Quem foi o gênio que inverteu o padrão RGB?
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
                    static_features, temporal_features = self._extract_features(hand_landmarks.landmark)
                else:
                    static_features = np.zeros(self.static_feature_dimension, dtype=np.float32)
                    temporal_features = np.zeros(self.temporal_feature_dimension, dtype=np.float32)

                self._append_temporal_features(temporal_features, timestamp)

                static_event = self._predict_static(static_features, timestamp)
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
