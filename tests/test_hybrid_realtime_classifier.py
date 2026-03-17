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


if __name__ == '__main__':
    unittest.main()
