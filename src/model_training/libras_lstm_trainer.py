"""
Treinador LSTM para Reconhecimento Temporal de Libras
====================================================

Treina um modelo recorrente sobre sequências de landmarks para sinais
dinâmicos como J e Z.
"""

import os
import pickle
from typing import Dict, Tuple

import numpy as np

from config.settings import LSTM_CONFIG, SEQUENCES_DIR

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    TENSORFLOW_AVAILABLE = False
    TF_IMPORT_ERROR = e


class LibrasLSTMTrainer:
    """Treina uma LSTM sobre sequências de landmarks."""

    def __init__(
        self,
        sequences_dir: str = SEQUENCES_DIR,
        model_path: str = LSTM_CONFIG['model_path'],
        label_map_path: str = LSTM_CONFIG['label_map_path'],
    ):
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError(
                "TensorFlow não disponível para treino LSTM. "
                f"Motivo: {type(TF_IMPORT_ERROR).__name__}: {TF_IMPORT_ERROR}"
            )

        self.sequences_dir = sequences_dir
        self.model_path = model_path
        self.label_map_path = label_map_path
        self.sequence_length = LSTM_CONFIG['sequence_length']
        self.feature_size = LSTM_CONFIG['feature_size']
        self.model = None
        self.label_map: Dict[int, str] = {}
        self.training_history: Dict[str, object] = {}

    def load_sequence_dataset(self) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
        """Carrega as sequências salvas em disco."""
        if not os.path.exists(self.sequences_dir):
            raise FileNotFoundError(f"Diretório de sequências não encontrado: {self.sequences_dir}")

        data = []
        labels = []
        self.label_map = {}

        class_dirs = [
            entry for entry in sorted(os.listdir(self.sequences_dir))
            if os.path.isdir(os.path.join(self.sequences_dir, entry))
        ]
        if not class_dirs:
            raise ValueError("Nenhuma classe de sequência encontrada")

        for idx, label in enumerate(class_dirs):
            self.label_map[idx] = label
            label_dir = os.path.join(self.sequences_dir, label)
            for filename in sorted(os.listdir(label_dir)):
                if not filename.endswith('.npy'):
                    continue

                sequence_path = os.path.join(label_dir, filename)
                sequence = np.load(sequence_path)

                if sequence.shape != (self.sequence_length, self.feature_size):
                    continue

                data.append(sequence.astype(np.float32))
                labels.append(idx)

        if not data:
            raise ValueError(
                "Nenhuma sequência válida encontrada com o shape esperado "
                f"({self.sequence_length}, {self.feature_size})"
            )

        return np.asarray(data), np.asarray(labels), self.label_map

    def build_model(self, num_classes: int):
        """Constrói a arquitetura LSTM."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.sequence_length, self.feature_size)),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
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
        epochs: int = LSTM_CONFIG['epochs'],
        batch_size: int = LSTM_CONFIG['batch_size'],
        validation_split: float = LSTM_CONFIG['validation_split'],
    ):
        """Treina a LSTM e persiste o modelo final."""
        if self.model is None:
            self.build_model(num_classes=len(np.unique(labels)))

        history = self.model.fit(
            data,
            labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
        )

        self.training_history = {
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_split': validation_split,
            'sequence_length': self.sequence_length,
            'feature_size': self.feature_size,
            'num_classes': len(np.unique(labels)),
            'classes': [self.label_map[idx] for idx in sorted(self.label_map)],
            'history': history.history,
        }
        self.save_model()
        return history

    def save_model(self):
        """Salva o modelo treinado e o mapa de labels."""
        if self.model is None:
            raise RuntimeError("Modelo ainda não foi treinado")

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)

        metadata = {
            'label_map': self.label_map,
            'training_history': self.training_history,
        }
        with open(self.label_map_path, 'wb') as file_obj:
            pickle.dump(metadata, file_obj)

        print(f"Modelo LSTM salvo em: {self.model_path}")
        print(f"Metadados salvos em: {self.label_map_path}")


def main():
    """Executa o treinamento LSTM usando as configurações padrão."""
    trainer = LibrasLSTMTrainer()
    data, labels, label_map = trainer.load_sequence_dataset()
    print(f"Sequências carregadas: {data.shape[0]}")
    print(f"Classes: {label_map}")
    trainer.train_model(data, labels)


if __name__ == '__main__':
    main()
