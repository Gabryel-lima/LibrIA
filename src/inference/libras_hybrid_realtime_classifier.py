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

from .sign_token import SignToken
from .temporal_pipeline import TemporalPipeline

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
        self.alphabet_dict = {i: chr(65 + i) for i in range(26)}
        self.static_features: Optional[np.ndarray] = None
        self.last_partial: Optional[SignToken] = None
        self.token_history: Deque[SignToken] = deque(
            maxlen=HYBRID_INFERENCE_CONFIG['overlay_history_size']
        )

        # O reconhecimento temporal deixou de ser janela fixa: o pipeline só
        # consulta o LSTM quando há um sinal delimitado por movimento, e o
        # modelo estático responde enquanto a mão está parada.
        self.pipeline = TemporalPipeline(
            temporal_predictor=self._predict_sequence,
            label_map=self.temporal_label_map,
            sequence_length=self.sequence_length,
            static_predictor=self._predict_static_features,
            config={'static_interval_frames': self.prediction_interval},
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

    def _predict_static_features(self, features: np.ndarray) -> Tuple[str, float]:
        """Prediz uma letra estática, testando também a versão espelhada (TTA).

        O pipeline entrega as features temporais. Quando os dois modelos usam
        modos de feature diferentes, usamos as features estáticas extraídas do
        mesmo quadro em vez de alimentar o modelo com o vetor errado.
        """
        if self.static_features is not None:
            features = self.static_features

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

        return self._static_prediction_to_token(prediction), confidence

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

    def _predict_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Prediz sobre uma sequência, testando também a versão espelhada (TTA).

        Devolve a distribuição de probabilidades — quem decide o rótulo é o
        pipeline, que ainda vai suavizá-la junto com as demais janelas.
        """
        sequence_array = np.asarray(sequence, dtype=np.float32)
        probabilities = self.temporal_model.predict(
            np.expand_dims(sequence_array, axis=0), verbose=0
        )[0]

        mirrored_probabilities = self.temporal_model.predict(
            np.expand_dims(self._mirror_sequence(sequence_array), axis=0),
            verbose=0,
        )[0]

        if float(np.max(mirrored_probabilities)) > float(np.max(probabilities)):
            return mirrored_probabilities
        return probabilities

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

        estado = 'SINALIZANDO' if self.pipeline.is_signing else 'aguardando'
        cv.putText(
            frame,
            f'{estado} | movimento: {self.pipeline.last_energy:.3f}',
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 0) if self.pipeline.is_signing else (200, 200, 200),
            2,
        )

        if self.last_partial is not None and self.pipeline.is_signing:
            cv.putText(
                frame,
                f'Parcial: {self.last_partial.label} {self.last_partial.confidence:.0%}',
                (10, 95),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 0),
                2,
            )

        if self.token_history:
            confirmados = ' '.join(
                token.label if not token.is_rejected else '?'
                for token in self.token_history
            )
            cv.putText(
                frame,
                f'Tokens: {confirmados}',
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

                hand_present = bool(results.multi_hand_landmarks)
                if hand_present:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )
                    self.static_features, temporal_features = self._extract_features(
                        hand_landmarks.landmark
                    )
                else:
                    self.static_features = None
                    temporal_features = None

                token = self.pipeline.process_frame(
                    temporal_features,
                    timestamp,
                    hand_present=hand_present,
                )

                if token is not None:
                    if token.state == 'partial':
                        self.last_partial = token
                    else:
                        self.token_history.append(token)
                        self.last_partial = None

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
