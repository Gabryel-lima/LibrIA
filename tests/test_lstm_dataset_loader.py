import os
import shutil
import tempfile
import unittest

import numpy as np

from src.model_training.libras_lstm_trainer import LibrasLSTMTrainer


class LSTMDatasetLoaderTests(unittest.TestCase):
    def test_load_sequence_dataset_adds_mirrored_sequences(self):
        project_root = os.getcwd()
        temp_dir = tempfile.mkdtemp(prefix='libria-lstm-loader-')

        try:
            os.chdir(temp_dir)
            os.makedirs('dataset/temporal/J', exist_ok=True)

            sequence = np.zeros((30, 21, 3), dtype=np.float32)
            sequence[:, :, 0] = np.linspace(0.1, 0.9, 21, dtype=np.float32)
            sequence[:, :, 1] = 0.5
            sequence[:, :, 2] = 0.25
            np.save('dataset/temporal/J/seq_000.npy', sequence)

            trainer = LibrasLSTMTrainer.__new__(LibrasLSTMTrainer)
            trainer.sequences_dir = 'dataset/temporal'
            trainer.sequence_length = 30
            trainer.feature_size = 63
            trainer.allowed_classes = ['J', 'Z']
            trainer.restrict_to_allowed_classes = False
            trainer.label_map = {}

            data, labels, label_map = trainer.load_sequence_dataset()

            self.assertEqual(data.shape, (2, 30, 63))
            self.assertEqual(labels.tolist(), [0, 0])
            self.assertEqual(label_map, {0: 'J'})

            original = data[0].reshape(30, 21, 3)
            mirrored = data[1].reshape(30, 21, 3)
            self.assertTrue(np.allclose(mirrored[:, :, 0], -original[:, :, 0]))
            self.assertTrue(np.allclose(mirrored[:, :, 1:], original[:, :, 1:]))
        finally:
            os.chdir(project_root)
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
