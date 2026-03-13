"""Treinador de CNN temporal quantizada para J e Z usando sequências em NPY."""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from config.settings import EMBEDDED_TEMPORAL_CONFIG, TEMPORAL_DATASET_DIR

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except (ImportError, RuntimeError) as error:
    TENSORFLOW_AVAILABLE = False
    TF_IMPORT_ERROR = error


def prepare_temporal_landmark_tensor(
    sequence: np.ndarray,
    sequence_length: int,
    feature_size: int,
) -> np.ndarray:
    """Normaliza uma sequência temporal para (sequence_length, feature_size)."""
    array = np.asarray(sequence, dtype=np.float32)
    if array.shape == (sequence_length, feature_size):
        return array
    if array.shape == (sequence_length, 21, 3):
        return array.reshape(sequence_length, feature_size)
    raise ValueError(f'Shape temporal incompatível para CNN embedded: {array.shape}')


class LibrasEmbeddedTemporalCNNTrainer:
    """Treina uma CNN temporal pequena para J e Z em deployment embedded."""

    def __init__(
        self,
        dataset_dir: str = TEMPORAL_DATASET_DIR,
        model_path: str = EMBEDDED_TEMPORAL_CONFIG['keras_model_path'],
        tflite_path: str = EMBEDDED_TEMPORAL_CONFIG['tflite_model_path'],
        label_map_path: str = EMBEDDED_TEMPORAL_CONFIG['label_map_path'],
    ):
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError(
                'TensorFlow não disponível para treino temporal embedded. '
                f'Motivo: {type(TF_IMPORT_ERROR).__name__}: {TF_IMPORT_ERROR}'
            )

        self.dataset_dir = dataset_dir
        self.model_path = model_path
        self.tflite_path = tflite_path
        self.label_map_path = label_map_path
        self.sequence_length = int(EMBEDDED_TEMPORAL_CONFIG['sequence_length'])
        self.feature_size = int(EMBEDDED_TEMPORAL_CONFIG['feature_size'])
        self.validation_split = float(EMBEDDED_TEMPORAL_CONFIG['validation_split'])
        self.allowed_classes = [label.upper() for label in EMBEDDED_TEMPORAL_CONFIG['allowed_classes']]
        self.model = None
        self.label_map: Dict[int, str] = {}
        self.training_history: Dict[str, object] = {}

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(f'Diretório temporal embedded não encontrado: {self.dataset_dir}')

        data: List[np.ndarray] = []
        labels: List[int] = []
        class_names = [
            entry for entry in sorted(os.listdir(self.dataset_dir))
            if os.path.isdir(os.path.join(self.dataset_dir, entry)) and entry.upper() in self.allowed_classes
        ]

        if not class_names:
            raise ValueError('Nenhuma classe temporal permitida encontrada para treino embedded')

        for class_name in class_names:
            class_index = len(self.label_map)
            self.label_map[class_index] = class_name
            class_dir = os.path.join(self.dataset_dir, class_name)
            sequence_files = [
                filename for filename in sorted(os.listdir(class_dir))
                if filename.startswith('seq_') and filename.endswith('.npy')
            ]

            for filename in sequence_files:
                sequence_path = os.path.join(class_dir, filename)
                sequence = np.load(sequence_path)
                try:
                    prepared = prepare_temporal_landmark_tensor(
                        sequence,
                        sequence_length=self.sequence_length,
                        feature_size=self.feature_size,
                    )
                except ValueError:
                    continue
                data.append(prepared)
                labels.append(class_index)

        if not data:
            raise ValueError('Nenhuma sequência temporal válida encontrada para treino embedded')

        return np.asarray(data, dtype=np.float32), np.asarray(labels, dtype=np.int32), self.label_map

    def build_model(self, num_classes: int):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.sequence_length, self.feature_size)),
            tf.keras.layers.Conv1D(24, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool1D(pool_size=2),
            tf.keras.layers.Conv1D(48, 3, padding='same', activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax'),
        ])
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        return self.model

    def train_model(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        epochs: int = EMBEDDED_TEMPORAL_CONFIG['epochs'],
        batch_size: int = EMBEDDED_TEMPORAL_CONFIG['batch_size'],
    ) -> Dict[str, float]:
        if self.model is None:
            self.build_model(num_classes=len(np.unique(labels)))

        x_train, x_test, y_train, y_test = train_test_split(
            data,
            labels,
            test_size=self.validation_split,
            shuffle=True,
            stratify=labels,
            random_state=42,
        )

        history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )

        predictions = self.model.predict(x_test, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        accuracy = float(accuracy_score(y_test, predicted_labels))
        report = classification_report(
            y_test,
            predicted_labels,
            target_names=[self.label_map[index] for index in sorted(self.label_map)],
            zero_division=0,
        )

        self.training_history = {
            'epochs': int(epochs),
            'batch_size': int(batch_size),
            'validation_split': self.validation_split,
            'sequence_length': self.sequence_length,
            'feature_size': self.feature_size,
            'num_classes': len(np.unique(labels)),
            'classes': [self.label_map[index] for index in sorted(self.label_map)],
            'accuracy': accuracy,
            'history': history.history,
            'classification_report': report,
        }

        self._save_model()
        self._export_tflite_int8(x_train)
        return {'accuracy': accuracy}

    def _representative_dataset(self, samples: np.ndarray):
        limit = min(100, len(samples))
        for sample in samples[:limit]:
            yield [np.expand_dims(sample.astype(np.float32), axis=0)]

    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)

        metadata = {
            'label_map': self.label_map,
            'training_history': self.training_history,
            'input_shape': [self.sequence_length, self.feature_size],
            'preprocessing': 'sequências de landmarks em seq_XXX.npy',
        }
        with open(self.label_map_path, 'w', encoding='utf-8') as file_obj:
            json.dump(metadata, file_obj, ensure_ascii=True, indent=2)

    def _export_tflite_int8(self, representative_data: np.ndarray):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: self._representative_dataset(representative_data)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()
        with open(self.tflite_path, 'wb') as file_obj:
            file_obj.write(tflite_model)


def main():
    trainer = LibrasEmbeddedTemporalCNNTrainer()
    data, labels, label_map = trainer.load_dataset()
    metrics = trainer.train_model(data, labels)
    print(f'Sequências carregadas: {len(data)}')
    print(f'Classes temporais embedded: {list(label_map.values())}')
    print(f'Input temporal embedded: ({trainer.sequence_length}, {trainer.feature_size})')
    print(f'Acurácia temporal embedded: {metrics["accuracy"]:.4f}')
    print(f'Modelo Keras salvo em: {trainer.model_path}')
    print(f'Modelo TFLite int8 salvo em: {trainer.tflite_path}')


if __name__ == '__main__':
    main()
