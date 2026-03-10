"""
Classificador Temporal LSTM para Libras
=======================================

Executa inferência em tempo real usando uma janela deslizante de landmarks.
"""

import os
import pickle
from typing import Dict, List, Optional

import cv2 as cv
import numpy as np

from config.settings import CAMERA_CONFIG, FEATURE_DIMENSION, FEATURE_MODE, LSTM_CONFIG
from utils.helpers import extract_landmarks_by_mode, load_camera_calibration, preprocess_frame

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


class LibrasLSTMRealtimeClassifier:
    """Executa inferência temporal com uma LSTM treinada."""

    def __init__(
        self,
        model_path: str = LSTM_CONFIG['model_path'],
        label_map_path: str = LSTM_CONFIG['label_map_path'],
        confidence_threshold: float = 0.85,
    ):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                "MediaPipe não disponível para inferência temporal. "
                f"Motivo: {type(MEDIAPIPE_ERROR).__name__}: {MEDIAPIPE_ERROR}"
            )
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError(
                "TensorFlow não disponível para inferência temporal. "
                f"Motivo: {type(TENSORFLOW_ERROR).__name__}: {TENSORFLOW_ERROR}"
            )

        self.model_path = model_path
        self.label_map_path = label_map_path
        self.confidence_threshold = confidence_threshold
        self.sequence_length = LSTM_CONFIG['sequence_length']
        self.feature_dimension = FEATURE_DIMENSION
        self.feature_mode = FEATURE_MODE
        self.sequence_buffer: List[np.ndarray] = []

        self.model = self._load_model()
        self.label_map = self._load_label_map()
        self.camera_calibration = self._load_camera_calibration()

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.5,
        )

        self.last_prediction: Optional[str] = None
        self.last_confidence = 0.0

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo LSTM não encontrado: {self.model_path}")
        return tf.keras.models.load_model(self.model_path)

    def _load_label_map(self) -> Dict[int, str]:
        if not os.path.exists(self.label_map_path):
            raise FileNotFoundError(f"Metadados de labels não encontrados: {self.label_map_path}")

        with open(self.label_map_path, 'rb') as file_obj:
            metadata = pickle.load(file_obj)
        return metadata.get('label_map', {})

    def _load_camera_calibration(self):
        if not CAMERA_CONFIG['enabled']:
            return None
        return load_camera_calibration(
            CAMERA_CONFIG['camera_matrix_path'],
            CAMERA_CONFIG['dist_coeffs_path'],
        )

    def _predict_sequence(self):
        if len(self.sequence_buffer) < self.sequence_length:
            return

        input_sequence = np.expand_dims(np.asarray(self.sequence_buffer, dtype=np.float32), axis=0)
        probabilities = self.model.predict(input_sequence, verbose=0)[0]
        label_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[label_idx])

        if confidence >= self.confidence_threshold:
            self.last_prediction = self.label_map.get(label_idx, str(label_idx))
            self.last_confidence = confidence

    def _append_features(self, features: np.ndarray):
        self.sequence_buffer.append(features)
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer.pop(0)

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        cv.putText(
            frame,
            'LibrIA - Inferencia Temporal LSTM',
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv.putText(
            frame,
            f'Buffer: {len(self.sequence_buffer)}/{self.sequence_length}',
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 200),
            2,
        )
        if self.last_prediction is not None:
            cv.putText(
                frame,
                f'Sinal: {self.last_prediction} ({self.last_confidence:.2%})',
                (10, 95),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )
        return frame

    def start_classification(self, camera_index: int = 0):
        """Inicia a inferência temporal em tempo real."""
        cap = cv.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera {camera_index}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = preprocess_frame(frame, self.camera_calibration)
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

                self._append_features(features)
                self._predict_sequence()
                frame = self._draw_overlay(frame)

                cv.imshow('LibrIA - Inferencia Temporal LSTM', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv.destroyAllWindows()
            self.hands.close()


def main():
    """Executa a inferência temporal com a configuração padrão."""
    classifier = LibrasLSTMRealtimeClassifier()
    classifier.start_classification()


if __name__ == '__main__':
    main()