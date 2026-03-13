"""Treinador de CNN quantizada para deployment embedded usando landmarks em NPY."""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from config.settings import EMBEDDED_CONFIG, STATIC_DATASET_DIR, STATIC_LABELS

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except (ImportError, RuntimeError) as error:
    TENSORFLOW_AVAILABLE = False
    TF_IMPORT_ERROR = error


def prepare_static_landmark_tensor(sample: np.ndarray) -> np.ndarray:
    """Normaliza o formato dos landmarks estáticos para (21, 3)."""
    array = np.asarray(sample, dtype=np.float32)
    if array.shape == (21, 3):
        return array
    if array.ndim == 1 and array.shape[0] == 63:
        return array.reshape(21, 3)
    raise ValueError(f'Shape de sample estático incompatível para CNN embedded: {array.shape}')


class LibrasEmbeddedCNNTrainer:
    """Treina uma CNN pequena sobre landmarks estáticos para deployment embedded."""

    def __init__(
        self,
        dataset_dir: str = STATIC_DATASET_DIR,
        model_path: str = EMBEDDED_CONFIG['keras_model_path'],
        tflite_path: str = EMBEDDED_CONFIG['tflite_model_path'],
        label_map_path: str = EMBEDDED_CONFIG['label_map_path'],
    ):
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError(
                'TensorFlow não disponível para treino embedded. '
                f'Motivo: {type(TF_IMPORT_ERROR).__name__}: {TF_IMPORT_ERROR}'
            )

        self.dataset_dir = dataset_dir
        self.model_path = model_path
        self.tflite_path = tflite_path
        self.label_map_path = label_map_path
        self.input_points = int(EMBEDDED_CONFIG['input_points'])
        self.input_channels = int(EMBEDDED_CONFIG['input_channels'])
        self.min_samples_per_class = int(EMBEDDED_CONFIG['min_samples_per_class'])
        self.validation_split = float(EMBEDDED_CONFIG['validation_split'])
        self.model = None
        self.label_map: Dict[int, str] = {}
        self.training_history: Dict[str, object] = {}

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
        """Carrega landmarks estáticos em NPY para treino embedded."""
        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(f'Diretório do dataset embedded não encontrado: {self.dataset_dir}')

        data: List[np.ndarray] = []
        labels: List[int] = []
        allowed_labels = set(STATIC_LABELS)
        class_names = [
            entry for entry in sorted(os.listdir(self.dataset_dir))
            if os.path.isdir(os.path.join(self.dataset_dir, entry)) and entry in allowed_labels
        ]

        if not class_names:
            raise ValueError('Nenhuma classe estática válida encontrada para treino embedded')

        for class_name in class_names:
            class_dir = os.path.join(self.dataset_dir, class_name)
            sample_files = [
                filename for filename in sorted(os.listdir(class_dir))
                if filename.startswith('sample_') and filename.endswith('.npy')
            ]

            if len(sample_files) < self.min_samples_per_class:
                continue

            class_index = len(self.label_map)
            self.label_map[class_index] = class_name

            for filename in sample_files:
                sample_path = os.path.join(class_dir, filename)
                sample = np.load(sample_path)
                try:
                    prepared = prepare_static_landmark_tensor(sample)
                except ValueError:
                    continue
                data.append(prepared)
                labels.append(class_index)

        if not data:
            raise ValueError(
                'Nenhum sample utilizável encontrado em dataset/static. '
                'O treino embedded espera arquivos sample_XXX.npy por classe.'
            )

        return np.asarray(data, dtype=np.float32), np.asarray(labels, dtype=np.int32), self.label_map

    def build_model(self, num_classes: int):
        """Constrói uma CNN pequena compatível com quantização int8."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_points, self.input_channels)),
            tf.keras.layers.Conv1D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv1D(24, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool1D(pool_size=2),
            tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu'),
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
        epochs: int = EMBEDDED_CONFIG['epochs'],
        batch_size: int = EMBEDDED_CONFIG['batch_size'],
    ) -> Dict[str, float]:
        """Treina o modelo e exporta artefatos Keras + TFLite int8."""
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
            'input_points': self.input_points,
            'input_channels': self.input_channels,
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
            'input_shape': [self.input_points, self.input_channels],
            'preprocessing': 'landmarks wrist_relative em sample_XXX.npy',
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
    trainer = LibrasEmbeddedCNNTrainer()
    data, labels, label_map = trainer.load_dataset()
    metrics = trainer.train_model(data, labels)
    print(f'Amostras carregadas: {len(data)}')
    print(f'Classes embedded: {list(label_map.values())}')
    print(f'Input embedded: ({trainer.input_points}, {trainer.input_channels})')
    print(f'Acurácia embedded: {metrics["accuracy"]:.4f}')
    print(f'Modelo Keras salvo em: {trainer.model_path}')
    print(f'Modelo TFLite int8 salvo em: {trainer.tflite_path}')


if __name__ == '__main__':
    main()