import os
import pickle
import shutil
import tempfile
import unittest

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.inference.libras_hybrid_realtime_classifier import LibrasHybridRealtimeClassifier


class HybridRealtimeClassifierStaticModelTests(unittest.TestCase):
    def setUp(self):
        self.project_root = os.getcwd()
        self.temp_dir = tempfile.mkdtemp(prefix='libria-hybrid-loader-')
        os.chdir(self.temp_dir)

    def tearDown(self):
        os.chdir(self.project_root)
        shutil.rmtree(self.temp_dir)

    def test_load_static_model_patches_legacy_estimators_and_infers_feature_mode(self):
        features = np.vstack(
            [
                np.zeros((4, 42), dtype=np.float32),
                np.ones((4, 42), dtype=np.float32),
            ]
        )
        labels = np.array(['0'] * 4 + ['1'] * 4)

        model = RandomForestClassifier(n_estimators=2, random_state=42)
        model.fit(features, labels)

        delattr(model, 'monotonic_cst')
        for estimator in model.estimators_:
            delattr(estimator, 'monotonic_cst')

        model_path = os.path.join(self.temp_dir, 'legacy_model.pickle')
        with open(model_path, 'wb') as file_obj:
            pickle.dump({'model': model}, file_obj)

        classifier = LibrasHybridRealtimeClassifier.__new__(LibrasHybridRealtimeClassifier)
        classifier.static_model_path = model_path

        loaded_model, metadata = classifier._load_static_model()

        self.assertIsNone(loaded_model.monotonic_cst)
        self.assertTrue(all(hasattr(estimator, 'monotonic_cst') for estimator in loaded_model.estimators_))
        self.assertTrue(all(estimator.monotonic_cst is None for estimator in loaded_model.estimators_))
        self.assertEqual(metadata['num_features'], 42)
        self.assertEqual(metadata['feature_mode'], 'bounding_box')

    def test_mirror_features_inverts_x_axis_for_landmark_layout(self):
        classifier = LibrasHybridRealtimeClassifier.__new__(LibrasHybridRealtimeClassifier)

        features = np.zeros(63, dtype=np.float32)
        features[0::3] = np.linspace(0.1, 0.9, 21, dtype=np.float32)
        features[1::3] = 0.5
        features[2::3] = 0.25

        mirrored = classifier._mirror_features(features)

        self.assertTrue(np.allclose(mirrored[0::3], -features[0::3]))
        self.assertTrue(np.allclose(mirrored[1::3], features[1::3]))
        self.assertTrue(np.allclose(mirrored[2::3], features[2::3]))

    def test_mirror_sequence_applies_framewise_mirroring(self):
        classifier = LibrasHybridRealtimeClassifier.__new__(LibrasHybridRealtimeClassifier)
        classifier.temporal_feature_mode = 'wrist_relative'
        classifier.sequence_length = 4

        sequence = np.zeros((4, 63), dtype=np.float32)
        sequence[:, 0::3] = np.linspace(0.2, 0.8, 21, dtype=np.float32)
        sequence[:, 1::3] = 0.5
        sequence[:, 2::3] = 0.25

        mirrored = classifier._mirror_sequence(sequence)

        self.assertTrue(np.allclose(mirrored[:, 0::3], -sequence[:, 0::3]))
        self.assertTrue(np.allclose(mirrored[:, 1::3], sequence[:, 1::3]))
        self.assertTrue(np.allclose(mirrored[:, 2::3], sequence[:, 2::3]))


if __name__ == '__main__':
    unittest.main()
