import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from scripts.collect_dataset import (
    CaptureContext,
    _backfill_static_samples_from_frames,
    _resolve_vocabulary_labels,
)
from config.vocabulary import MODALITY_STATIC, MODALITY_TEMPORAL, UNKNOWN_LABEL
from src.dataset.sample_metadata import read_metadata


class StaticCollectionBackfillTests(unittest.TestCase):
    def test_backfill_generates_missing_npy_from_frames(self):
        temp_dir = tempfile.mkdtemp(prefix='libria-static-backfill-')

        try:
            label_dir = os.path.join(temp_dir, 'A')
            os.makedirs(label_dir, exist_ok=True)

            frame = np.full((32, 32, 3), 255, dtype=np.uint8)
            frame_path = os.path.join(label_dir, 'frame_000.png')
            cv2.imwrite(frame_path, frame)

            generated = _backfill_static_samples_from_frames(
                label_dir,
                calibration=None,
                extractor=lambda image: np.ones((21, 3), dtype=np.float32),
            )

            self.assertEqual(generated, 1)
            self.assertTrue(os.path.exists(os.path.join(label_dir, 'sample_000.npy')))
            sample = np.load(os.path.join(label_dir, 'sample_000.npy'))
            self.assertEqual(sample.shape, (21, 3))

        finally:
            shutil.rmtree(temp_dir)

    def test_backfill_keeps_existing_npy_untouched(self):
        temp_dir = tempfile.mkdtemp(prefix='libria-static-backfill-')

        try:
            label_dir = os.path.join(temp_dir, 'A')
            os.makedirs(label_dir, exist_ok=True)

            frame = np.full((32, 32, 3), 200, dtype=np.uint8)
            cv2.imwrite(os.path.join(label_dir, 'frame_001.png'), frame)
            existing_sample = np.zeros((21, 3), dtype=np.float32)
            np.save(os.path.join(label_dir, 'sample_001.npy'), existing_sample)

            generated = _backfill_static_samples_from_frames(
                label_dir,
                calibration=None,
                extractor=lambda image: np.ones((21, 3), dtype=np.float32),
            )

            self.assertEqual(generated, 0)
            sample = np.load(os.path.join(label_dir, 'sample_001.npy'))
            self.assertTrue(np.array_equal(sample, existing_sample))

        finally:
            shutil.rmtree(temp_dir)

    def test_backfill_generates_mirrored_sample_when_enabled(self):
        temp_dir = tempfile.mkdtemp(prefix='libria-static-backfill-')

        try:
            label_dir = os.path.join(temp_dir, 'A')
            os.makedirs(label_dir, exist_ok=True)

            frame = np.full((32, 32, 3), 180, dtype=np.uint8)
            cv2.imwrite(os.path.join(label_dir, 'frame_002.png'), frame)

            sample = np.zeros((21, 3), dtype=np.float32)
            sample[:, 0] = np.linspace(0.1, 0.9, 21, dtype=np.float32)
            sample[:, 1] = 0.5
            sample[:, 2] = 0.25

            generated = _backfill_static_samples_from_frames(
                label_dir,
                calibration=None,
                extractor=lambda image: sample,
                save_mirrored=True,
            )

            self.assertEqual(generated, 2)
            self.assertTrue(os.path.exists(os.path.join(label_dir, 'sample_002.npy')))
            self.assertTrue(os.path.exists(os.path.join(label_dir, 'sample_002_mirror.npy')))

            mirrored = np.load(os.path.join(label_dir, 'sample_002_mirror.npy'))
            self.assertTrue(np.allclose(mirrored[:, 0], -sample[:, 0]))
            self.assertTrue(np.allclose(mirrored[:, 1:], sample[:, 1:]))
        finally:
            shutil.rmtree(temp_dir)


class StaticCollectionMetadataTests(unittest.TestCase):
    def test_backfill_writes_metadata_for_sample_and_mirror(self):
        temp_dir = tempfile.mkdtemp(prefix='libria-static-metadata-')

        try:
            label_dir = os.path.join(temp_dir, 'A')
            os.makedirs(label_dir, exist_ok=True)

            cv2.imwrite(
                os.path.join(label_dir, 'frame_003.png'),
                np.full((32, 32, 3), 160, dtype=np.uint8),
            )

            context = CaptureContext(
                subject_id='pessoa_01',
                camera_id='webcam_c920',
                environment='sala_luz_natural',
                dominant_hand='right',
            )

            generated = _backfill_static_samples_from_frames(
                label_dir,
                calibration=None,
                extractor=lambda image: np.ones((21, 3), dtype=np.float32),
                save_mirrored=True,
                context=context,
            )

            self.assertEqual(generated, 2)

            metadata = read_metadata(os.path.join(label_dir, 'sample_003.npy'))
            self.assertIsNotNone(metadata)
            self.assertEqual(metadata.label, 'A')
            self.assertEqual(metadata.modality, MODALITY_STATIC)
            self.assertEqual(metadata.sign_type, 'alphabet')
            self.assertEqual(metadata.subject_id, 'pessoa_01')
            self.assertEqual(metadata.camera_id, 'webcam_c920')
            self.assertEqual(metadata.environment, 'sala_luz_natural')
            self.assertEqual(metadata.dominant_hand, 'right')
            self.assertFalse(metadata.mirrored)

            mirrored = read_metadata(os.path.join(label_dir, 'sample_003_mirror.npy'))
            self.assertIsNotNone(mirrored)
            self.assertTrue(mirrored.mirrored)
            self.assertEqual(mirrored.dominant_hand, 'left')
            self.assertEqual(mirrored.source_sample, 'sample_003.npy')
        finally:
            shutil.rmtree(temp_dir)

    def test_backfill_without_context_keeps_legacy_behavior(self):
        temp_dir = tempfile.mkdtemp(prefix='libria-static-metadata-')

        try:
            label_dir = os.path.join(temp_dir, 'A')
            os.makedirs(label_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(label_dir, 'frame_004.png'),
                np.full((32, 32, 3), 160, dtype=np.uint8),
            )

            _backfill_static_samples_from_frames(
                label_dir,
                calibration=None,
                extractor=lambda image: np.ones((21, 3), dtype=np.float32),
            )

            self.assertTrue(os.path.exists(os.path.join(label_dir, 'sample_004.npy')))
            self.assertIsNone(read_metadata(os.path.join(label_dir, 'sample_004.npy')))
        finally:
            shutil.rmtree(temp_dir)


class VocabularySelectionTests(unittest.TestCase):
    def test_temporal_selection_covers_each_family(self):
        alphabet = _resolve_vocabulary_labels('alphabet', MODALITY_TEMPORAL)
        lexical = _resolve_vocabulary_labels('lexical', MODALITY_TEMPORAL)
        functional = _resolve_vocabulary_labels('functional', MODALITY_TEMPORAL)
        every = _resolve_vocabulary_labels('all', MODALITY_TEMPORAL)

        self.assertEqual(alphabet, ['J', 'Z'])
        self.assertIn('OI', lexical)
        self.assertIn('APAGAR', functional)
        self.assertEqual(set(every), set(alphabet) | set(lexical) | set(functional))

    def test_unknown_selection_targets_the_rejection_class(self):
        self.assertEqual(
            _resolve_vocabulary_labels('unknown', MODALITY_TEMPORAL), [UNKNOWN_LABEL]
        )

    def test_static_selection_stays_on_the_alphabet(self):
        labels = _resolve_vocabulary_labels('all', MODALITY_STATIC)

        self.assertIn('A', labels)
        self.assertNotIn('J', labels)
        self.assertNotIn('OI', labels)


if __name__ == '__main__':
    unittest.main()
