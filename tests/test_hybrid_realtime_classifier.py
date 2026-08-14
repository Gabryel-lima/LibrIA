import os
import pickle
import shutil
import tempfile
import unittest

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.inference.libras_hybrid_realtime_classifier import LibrasHybridRealtimeClassifier
from src.inference.temporal_pipeline import TemporalPipeline


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


class _FakeTemporalModel:
    """Modelo Keras mínimo: sempre a mesma distribuição."""

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=np.float32)
        self.calls = []

    def predict(self, batch, verbose=0):
        self.calls.append(np.asarray(batch))
        return np.expand_dims(self.probabilities, axis=0)


class _FakeStaticModel:
    def __init__(self, label, confidence):
        self.label = label
        self.confidence = confidence

    def predict(self, batch):
        return np.array([self.label])

    def predict_proba(self, batch):
        return np.array([[self.confidence, 1.0 - self.confidence]])


class HybridClassifierPipelineWiringTests(unittest.TestCase):
    """Garante que o classificador alimenta o pipeline temporal corretamente."""

    def _classifier(self):
        classifier = LibrasHybridRealtimeClassifier.__new__(LibrasHybridRealtimeClassifier)
        classifier.sequence_length = 4
        classifier.temporal_feature_mode = 'wrist_relative'
        classifier.temporal_label_map = {0: 'OI', 1: 'SIM'}
        classifier.temporal_model = _FakeTemporalModel([0.95, 0.05])
        classifier.static_model = _FakeStaticModel('A', 0.96)
        classifier.static_feature_mode = 'wrist_relative'
        classifier.static_features = None
        classifier.alphabet_dict = {index: chr(65 + index) for index in range(26)}

        classifier.pipeline = TemporalPipeline(
            temporal_predictor=classifier._predict_sequence,
            label_map=classifier.temporal_label_map,
            sequence_length=classifier.sequence_length,
            static_predictor=classifier._predict_static_features,
            config={
                'motion_smoothing': 0.0,
                'motion_start_threshold': 0.05,
                'motion_end_threshold': 0.02,
                'min_start_frames': 2,
                'min_end_frames': 2,
                'min_segment_frames': 3,
                'min_duration_seconds': 0.0,
                'emit_partial_tokens': False,
                'static_interval_frames': 1,
                'duplicate_window_seconds': 1.0,
            },
        )
        return classifier

    def test_sequence_prediction_returns_a_probability_distribution(self):
        classifier = self._classifier()

        probabilities = classifier._predict_sequence(np.zeros((4, 63), dtype=np.float32))

        self.assertEqual(len(probabilities), 2)
        self.assertAlmostEqual(float(np.max(probabilities)), 0.95, places=5)
        # Original e espelhada: duas passagens pelo modelo (TTA).
        self.assertEqual(len(classifier.temporal_model.calls), 2)

    def test_moving_hand_produces_a_temporal_token_through_the_pipeline(self):
        classifier = self._classifier()

        token = None
        position = 0.0
        for index in range(6):
            position += 0.1
            token = classifier.pipeline.process_frame(
                np.full(63, position, dtype=np.float32), index * 0.1
            ) or token
        for index in range(6, 10):
            token = classifier.pipeline.process_frame(
                np.full(63, position, dtype=np.float32), index * 0.1
            ) or token

        self.assertIsNotNone(token)
        self.assertEqual(token.label, 'OI')
        self.assertEqual(token.source, 'temporal')
        self.assertTrue(token.is_final)

    def test_still_hand_falls_back_to_the_static_model(self):
        classifier = self._classifier()

        token = None
        for index in range(6):
            token = classifier.pipeline.process_frame(
                np.full(63, 0.5, dtype=np.float32), index * 0.1
            ) or token

        self.assertIsNotNone(token)
        self.assertEqual(token.label, 'A')
        self.assertEqual(token.source, 'static')

    def test_static_predictor_uses_static_features_when_modes_differ(self):
        classifier = self._classifier()
        classifier.static_features = np.full(42, 0.2, dtype=np.float32)

        captured = {}
        original_predict = classifier.static_model.predict

        def capture(batch):
            captured['size'] = np.asarray(batch).shape[-1]
            return original_predict(batch)

        classifier.static_model.predict = capture
        classifier._predict_static_features(np.full(63, 0.9, dtype=np.float32))

        self.assertEqual(captured['size'], 42)


if __name__ == '__main__':
    unittest.main()
