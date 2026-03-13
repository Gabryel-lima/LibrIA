import unittest

import numpy as np

from src.model_training.libras_embedded_cnn_trainer import prepare_static_landmark_tensor
from src.model_training.libras_embedded_temporal_cnn_trainer import prepare_temporal_landmark_tensor


class EmbeddedCNNTensorPreparationTests(unittest.TestCase):
    def test_prepare_static_landmark_tensor_preserves_21x3(self):
        sample = np.arange(63, dtype=np.float32).reshape(21, 3)

        tensor = prepare_static_landmark_tensor(sample)

        self.assertEqual(tensor.shape, (21, 3))
        self.assertEqual(tensor.dtype, np.float32)

    def test_prepare_static_landmark_tensor_accepts_flattened_vector(self):
        sample = np.arange(63, dtype=np.float32)

        tensor = prepare_static_landmark_tensor(sample)

        self.assertEqual(tensor.shape, (21, 3))

    def test_prepare_temporal_landmark_tensor_accepts_30x21x3(self):
        sample = np.zeros((30, 21, 3), dtype=np.float32)

        tensor = prepare_temporal_landmark_tensor(sample, sequence_length=30, feature_size=63)

        self.assertEqual(tensor.shape, (30, 63))

    def test_prepare_temporal_landmark_tensor_accepts_30x63(self):
        sample = np.zeros((30, 63), dtype=np.float32)

        tensor = prepare_temporal_landmark_tensor(sample, sequence_length=30, feature_size=63)

        self.assertEqual(tensor.shape, (30, 63))


if __name__ == '__main__':
    unittest.main()