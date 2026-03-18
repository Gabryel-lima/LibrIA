import os
import shutil
import tempfile
import unittest

import numpy as np

from src.model_training.libras_model_trainer import LibrasModelTrainer


class StaticDatasetLoaderTests(unittest.TestCase):
    def test_load_dataset_from_static_directory(self):
        project_root = os.getcwd()
        temp_dir = tempfile.mkdtemp(prefix='libria-static-loader-')

        try:
            os.chdir(temp_dir)
            os.makedirs('dataset/static/A', exist_ok=True)
            np.save('dataset/static/A/sample_000.npy', np.zeros((21, 3), dtype=np.float32))

            trainer = LibrasModelTrainer(dataset_path='dataset/missing.pickle', model_output_dir='model')
            trainer.static_dataset_dir = 'dataset/static'

            data, labels = trainer.load_dataset()

            self.assertEqual(data.shape, (2, 63))
            self.assertEqual(labels.tolist(), ['A', 'A'])
        finally:
            os.chdir(project_root)
            shutil.rmtree(temp_dir)

    def test_load_dataset_adds_mirrored_static_samples(self):
        project_root = os.getcwd()
        temp_dir = tempfile.mkdtemp(prefix='libria-static-loader-')

        try:
            os.chdir(temp_dir)
            os.makedirs('dataset/static/A', exist_ok=True)

            sample = np.zeros((21, 3), dtype=np.float32)
            sample[:, 0] = np.linspace(0.05, 0.95, 21, dtype=np.float32)
            sample[:, 1] = 0.5
            sample[:, 2] = 0.25
            np.save('dataset/static/A/sample_000.npy', sample)

            trainer = LibrasModelTrainer(dataset_path='dataset/missing.pickle', model_output_dir='model')
            trainer.static_dataset_dir = 'dataset/static'

            data, labels = trainer.load_dataset()

            self.assertEqual(data.shape, (2, 63))
            self.assertEqual(labels.tolist(), ['A', 'A'])
            original = data[0].reshape(21, 3)
            mirrored = data[1].reshape(21, 3)
            self.assertTrue(np.allclose(mirrored[:, 0], -original[:, 0]))
            self.assertTrue(np.allclose(mirrored[:, 1:], original[:, 1:]))
        finally:
            os.chdir(project_root)
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
