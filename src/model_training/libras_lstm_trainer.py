"""
Treinador LSTM para Reconhecimento Temporal de Libras
====================================================

Treina um modelo recorrente sobre sequências de landmarks para sinais
dinâmicos como J e Z.
"""

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

from config.settings import FEATURE_MODE, LEGACY_TEMPORAL_DATASET_DIR, LSTM_CONFIG, TEMPORAL_DATASET_DIR
from src.utils.plots import plot_training_history

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
        sequences_dir: str = TEMPORAL_DATASET_DIR,
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
        self.allowed_classes: List[str] = list(LSTM_CONFIG.get('allowed_classes', []))
        self.restrict_to_allowed_classes = bool(LSTM_CONFIG.get('restrict_to_allowed_classes', False))
        self.model = None
        self.label_map: Dict[int, str] = {}
        self.training_history: Dict[str, object] = {}

    def _resolve_sequences_dir(self) -> str:
        """Retorna o diretório temporal atual com fallback para o legado."""
        # Se o diretório configurado existe, verifique se contém classes/arquivos
        if os.path.isdir(self.sequences_dir):
            # retornar se houver subdiretórios (classes) ou arquivos .npy
            try:
                entries = [e for e in os.listdir(self.sequences_dir) if e and not e.startswith('.')]
            except OSError:
                entries = []

            if entries:
                return self.sequences_dir

        # Fallback para o diretório legado, caso haja dados lá
        if os.path.isdir(LEGACY_TEMPORAL_DATASET_DIR):
            try:
                legacy_entries = [e for e in os.listdir(LEGACY_TEMPORAL_DATASET_DIR) if e and not e.startswith('.')]
            except OSError:
                legacy_entries = []

            if legacy_entries:
                return LEGACY_TEMPORAL_DATASET_DIR

        # Por fim, retornar o configurado (mesmo que esteja vazio)
        return self.sequences_dir

    def _get_allowed_class_set(self) -> set:
        return {label.upper() for label in self.allowed_classes}

    def _filter_class_dirs(self, class_dirs: List[str]) -> List[str]:
        if not self.restrict_to_allowed_classes:
            return class_dirs

        allowed_classes = self._get_allowed_class_set()
        filtered_dirs = [entry for entry in class_dirs if entry.upper() in allowed_classes]
        missing_classes = sorted(allowed_classes.difference({entry.upper() for entry in filtered_dirs}))

        if missing_classes:
            raise ValueError(
                'Classes temporais obrigatórias não encontradas no dataset: '
                f"{', '.join(missing_classes)}"
            )

        if not filtered_dirs:
            raise ValueError('Nenhuma classe temporal permitida encontrada para treino')

        return filtered_dirs

    def load_sequence_dataset(self) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
        """Carrega as sequências salvas em disco."""
        sequences_dir = self._resolve_sequences_dir()
        if not os.path.exists(sequences_dir):
            raise FileNotFoundError(f"Diretório de sequências não encontrado: {sequences_dir}")

        data = []
        labels = []
        self.label_map = {}

        class_dirs = [
            entry for entry in sorted(os.listdir(sequences_dir))
            if os.path.isdir(os.path.join(sequences_dir, entry))
        ]
        class_dirs = self._filter_class_dirs(class_dirs)
        if not class_dirs:
            raise ValueError("Nenhuma classe de sequência encontrada")

        for idx, label in enumerate(class_dirs):
            self.label_map[idx] = label
            label_dir = os.path.join(sequences_dir, label)
            for filename in sorted(os.listdir(label_dir)):
                if not filename.endswith('.npy'):
                    continue

                sequence_path = os.path.join(label_dir, filename)
                sequence = np.load(sequence_path)

                if sequence.ndim == 3:
                    sequence = sequence.reshape(sequence.shape[0], -1)

                if sequence.shape != (self.sequence_length, self.feature_size):
                    continue

                sequence = sequence.astype(np.float32)
                data.append(sequence)
                labels.append(idx)

                # Evita duplicar espelhamento quando o arquivo já é _mirror.
                if '_mirror' in filename:
                    continue

                data.append(self._mirror_sequence(sequence))
                labels.append(idx)

        if not data:
            raise ValueError(
                "Nenhuma sequência válida encontrada com o shape esperado "
                f"({self.sequence_length}, {self.feature_size})"
            )

        return np.asarray(data), np.asarray(labels), self.label_map

    def _mirror_sequence(self, sequence: np.ndarray) -> np.ndarray:
        mirrored = np.asarray(sequence, dtype=np.float32).copy()
        if mirrored.ndim != 2:
            mirrored = mirrored.reshape(self.sequence_length, self.feature_size)

        if self.feature_size % 3 != 0:
            return mirrored

        if FEATURE_MODE == 'wrist_relative':
            mirrored[:, 0::3] = -mirrored[:, 0::3]
        else:
            mirrored[:, 0::3] = 1.0 - mirrored[:, 0::3]
        return mirrored

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
            'allowed_classes': self.allowed_classes,
            'restrict_to_allowed_classes': self.restrict_to_allowed_classes,
            'history': history.history,
        }
        # salvar gráficos de treinamento (histórico + gráfico de precisão separado)
        try:
            plot_training_history({
                'training_loss': history.history.get('loss', []),
                'validation_loss': history.history.get('val_loss', []),
                'training_accuracy': history.history.get('accuracy', []),
                'validation_accuracy': history.history.get('val_accuracy', []),
            }, save_path=os.path.join('training_plots', 'training_history_lstm.png'))
        except Exception:
            pass

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
            'allowed_classes': self.allowed_classes,
            'restrict_to_allowed_classes': self.restrict_to_allowed_classes,
            'sequence_length': self.sequence_length,
            'feature_size': self.feature_size,
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
