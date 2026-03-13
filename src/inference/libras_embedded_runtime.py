"""Runtime híbrido para verificar o bundle embedded com modelos TFLite quantizados."""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from config.settings import EMBEDDED_BUNDLE_CONFIG, EMBEDDED_CONFIG, EMBEDDED_TEMPORAL_CONFIG
from src.model_training.libras_embedded_cnn_trainer import prepare_static_landmark_tensor
from src.model_training.libras_embedded_temporal_cnn_trainer import prepare_temporal_landmark_tensor

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except (ImportError, RuntimeError) as error:
    TENSORFLOW_AVAILABLE = False
    TF_IMPORT_ERROR = error


@dataclass(frozen=True)
class EmbeddedPrediction:
    token: str
    confidence: float
    source: str


def choose_embedded_prediction(
    static_prediction: Optional[EmbeddedPrediction],
    temporal_prediction: Optional[EmbeddedPrediction],
    static_threshold: float,
    temporal_threshold: float,
    temporal_priority_classes: List[str],
) -> Optional[EmbeddedPrediction]:
    priority_tokens = {token.upper() for token in temporal_priority_classes}

    if temporal_prediction is not None:
        if (
            temporal_prediction.token.upper() in priority_tokens
            and temporal_prediction.confidence >= temporal_threshold
        ):
            return temporal_prediction

    if static_prediction is not None and static_prediction.confidence >= static_threshold:
        return static_prediction

    return None


class QuantizedTFLiteModel:
    """Wrapper mínimo para modelos TFLite int8."""

    def __init__(self, model_path: str):
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError(
                'TensorFlow não disponível para inferência embedded. '
                f'Motivo: {type(TF_IMPORT_ERROR).__name__}: {TF_IMPORT_ERROR}'
            )

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

    def _quantize_input(self, tensor: np.ndarray) -> np.ndarray:
        scale, zero_point = self.input_details['quantization']
        if scale == 0:
            return tensor.astype(self.input_details['dtype'])

        quantized = np.round(tensor / scale + zero_point)
        if np.issubdtype(self.input_details['dtype'], np.integer):
            dtype_info = np.iinfo(self.input_details['dtype'])
            quantized = np.clip(quantized, dtype_info.min, dtype_info.max)
        return quantized.astype(self.input_details['dtype'])

    def _dequantize_output(self, tensor: np.ndarray) -> np.ndarray:
        scale, zero_point = self.output_details['quantization']
        if scale == 0:
            return tensor.astype(np.float32)
        return (tensor.astype(np.float32) - zero_point) * scale

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        batch = np.expand_dims(np.asarray(tensor, dtype=np.float32), axis=0)
        self.interpreter.set_tensor(self.input_details['index'], self._quantize_input(batch))
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details['index'])
        probabilities = self._dequantize_output(output)[0]
        if probabilities.sum() <= 0:
            return probabilities
        return probabilities / probabilities.sum()


class LibrasEmbeddedRuntime:
    """Combina os modelos embedded estático e temporal em um runtime único."""

    def __init__(self, manifest_path: str = EMBEDDED_BUNDLE_CONFIG['manifest_path']):
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f'Bundle embedded não encontrado: {manifest_path}. Execute o export embedded primeiro.'
            )

        with open(manifest_path, 'r', encoding='utf-8') as file_obj:
            self.manifest = json.load(file_obj)

        self.bundle_dir = os.path.dirname(manifest_path)
        self.static_model = QuantizedTFLiteModel(
            os.path.join(self.bundle_dir, self.manifest['static']['model_file'])
        )
        self.temporal_model = QuantizedTFLiteModel(
            os.path.join(self.bundle_dir, self.manifest['temporal']['model_file'])
        )
        self.static_labels = list(self.manifest['static']['labels'])
        self.temporal_labels = list(self.manifest['temporal']['labels'])
        self.static_threshold = float(self.manifest['hybrid']['static_confidence_threshold'])
        self.temporal_threshold = float(self.manifest['hybrid']['temporal_confidence_threshold'])
        self.temporal_priority_classes = list(self.manifest['hybrid']['temporal_priority_classes'])

    def predict_static(self, sample: np.ndarray) -> EmbeddedPrediction:
        prepared = prepare_static_landmark_tensor(sample)
        probabilities = self.static_model.predict(prepared)
        index = int(np.argmax(probabilities))
        return EmbeddedPrediction(
            token=self.static_labels[index],
            confidence=float(probabilities[index]),
            source='static',
        )

    def predict_temporal(self, sequence: np.ndarray) -> EmbeddedPrediction:
        input_shape = self.manifest['temporal']['input_shape']
        prepared = prepare_temporal_landmark_tensor(
            sequence,
            sequence_length=int(input_shape[0]),
            feature_size=int(input_shape[1]),
        )
        probabilities = self.temporal_model.predict(prepared)
        index = int(np.argmax(probabilities))
        return EmbeddedPrediction(
            token=self.temporal_labels[index],
            confidence=float(probabilities[index]),
            source='temporal',
        )

    def predict_hybrid(
        self,
        static_sample: Optional[np.ndarray] = None,
        temporal_sequence: Optional[np.ndarray] = None,
    ) -> Optional[EmbeddedPrediction]:
        static_prediction = self.predict_static(static_sample) if static_sample is not None else None
        temporal_prediction = self.predict_temporal(temporal_sequence) if temporal_sequence is not None else None
        return choose_embedded_prediction(
            static_prediction=static_prediction,
            temporal_prediction=temporal_prediction,
            static_threshold=self.static_threshold,
            temporal_threshold=self.temporal_threshold,
            temporal_priority_classes=self.temporal_priority_classes,
        )

    def evaluate_datasets(
        self,
        static_dataset_dir: str = EMBEDDED_CONFIG['dataset_dir'],
        temporal_dataset_dir: str = EMBEDDED_TEMPORAL_CONFIG['dataset_dir'],
    ) -> Dict[str, object]:
        static_total = 0
        static_correct = 0
        temporal_total = 0
        temporal_correct = 0
        hybrid_total = 0
        hybrid_correct = 0

        for label in sorted(self.static_labels):
            label_dir = os.path.join(static_dataset_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for filename in sorted(os.listdir(label_dir)):
                if not filename.startswith('sample_') or not filename.endswith('.npy'):
                    continue
                sample = np.load(os.path.join(label_dir, filename))
                prediction = self.predict_static(sample)
                static_total += 1
                hybrid_total += 1
                if prediction.token.upper() == label.upper():
                    static_correct += 1
                    hybrid_correct += 1

        for label in sorted(self.temporal_labels):
            label_dir = os.path.join(temporal_dataset_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for filename in sorted(os.listdir(label_dir)):
                if not filename.startswith('seq_') or not filename.endswith('.npy'):
                    continue
                sequence = np.load(os.path.join(label_dir, filename))
                temporal_prediction = self.predict_temporal(sequence)
                hybrid_prediction = self.predict_hybrid(temporal_sequence=sequence)
                temporal_total += 1
                hybrid_total += 1
                if temporal_prediction.token.upper() == label.upper():
                    temporal_correct += 1
                if hybrid_prediction is not None and hybrid_prediction.token.upper() == label.upper():
                    hybrid_correct += 1

        def _safe_accuracy(correct: int, total: int) -> float:
            return float(correct / total) if total else 0.0

        return {
            'static_samples': static_total,
            'static_accuracy': _safe_accuracy(static_correct, static_total),
            'temporal_sequences': temporal_total,
            'temporal_accuracy': _safe_accuracy(temporal_correct, temporal_total),
            'hybrid_total': hybrid_total,
            'hybrid_accuracy': _safe_accuracy(hybrid_correct, hybrid_total),
            'temporal_priority_classes': list(self.temporal_priority_classes),
            'landmark_contract': self.manifest['landmark_contract'],
        }


def main():
    runtime = LibrasEmbeddedRuntime()
    metrics = runtime.evaluate_datasets()
    print('=== LibrIA Embedded Runtime Check ===')
    print(f"Amostras estaticas: {metrics['static_samples']}")
    print(f"Acuracia estatica: {metrics['static_accuracy']:.2%}")
    print(f"Sequencias temporais: {metrics['temporal_sequences']}")
    print(f"Acuracia temporal: {metrics['temporal_accuracy']:.2%}")
    print(f"Acuracia hibrida: {metrics['hybrid_accuracy']:.2%}")
    print(f"Classes temporais prioritarias: {metrics['temporal_priority_classes']}")
    print('Contrato de landmarks:')
    print(metrics['landmark_contract']['note'])


if __name__ == '__main__':
    main()