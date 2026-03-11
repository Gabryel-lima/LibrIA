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

            self.assertEqual(data.shape, (1, 63))
            self.assertEqual(labels.tolist(), ['A'])
        finally:
            os.chdir(project_root)
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()