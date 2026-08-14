import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from src.dataset.sample_metadata import (
    HAND_LEFT,
    HAND_RIGHT,
    METADATA_SCHEMA_VERSION,
    UNSPECIFIED,
    SampleMetadata,
    collect_dataset_metadata,
    metadata_coverage,
    metadata_path_for,
    mirror_hand,
    read_metadata,
    write_metadata,
)


def _write_sample(label_dir, name, metadata=None):
    os.makedirs(label_dir, exist_ok=True)
    sample_path = os.path.join(label_dir, name)
    np.save(sample_path, np.zeros((21, 3), dtype=np.float32))
    if metadata is not None:
        write_metadata(sample_path, metadata)
    return sample_path


class SampleMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='libria-metadata-')

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_metadata_path_is_json_sibling_of_sample(self):
        self.assertEqual(
            metadata_path_for(os.path.join('dataset', 'static', 'A', 'sample_000.npy')),
            os.path.join('dataset', 'static', 'A', 'sample_000.json'),
        )

    def test_write_and_read_roundtrip_preserves_fields(self):
        metadata = SampleMetadata(
            label='A',
            modality='static',
            sign_type='alphabet',
            subject_id='pessoa_01',
            camera_id='webcam_c920',
            environment='sala_luz_natural',
            dominant_hand='right',
            capture_hand='Right',
            duration_seconds=1.5,
            quality=0.92,
            feature_mode='wrist_relative',
            feature_dimension=63,
        )
        sample_path = _write_sample(os.path.join(self.temp_dir, 'A'), 'sample_000.npy', metadata)

        loaded = read_metadata(sample_path)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.label, 'A')
        self.assertEqual(loaded.subject_id, 'pessoa_01')
        self.assertEqual(loaded.environment, 'sala_luz_natural')
        self.assertEqual(loaded.capture_hand, HAND_RIGHT)
        self.assertEqual(loaded.duration_seconds, 1.5)
        self.assertEqual(loaded.quality, 0.92)
        self.assertEqual(loaded.schema_version, METADATA_SCHEMA_VERSION)
        self.assertFalse(loaded.mirrored)

    def test_missing_metadata_returns_none(self):
        sample_path = _write_sample(os.path.join(self.temp_dir, 'A'), 'sample_000.npy')
        self.assertIsNone(read_metadata(sample_path))

    def test_corrupt_metadata_returns_none_instead_of_raising(self):
        sample_path = _write_sample(os.path.join(self.temp_dir, 'A'), 'sample_000.npy')
        with open(metadata_path_for(sample_path), 'w', encoding='utf-8') as file_obj:
            file_obj.write('{ not json')

        self.assertIsNone(read_metadata(sample_path))

    def test_unknown_fields_are_ignored_on_read(self):
        sample_path = _write_sample(os.path.join(self.temp_dir, 'A'), 'sample_000.npy')
        with open(metadata_path_for(sample_path), 'w', encoding='utf-8') as file_obj:
            json.dump({'label': 'A', 'modality': 'static', 'campo_futuro': 1}, file_obj)

        loaded = read_metadata(sample_path)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.label, 'A')

    def test_hand_normalization_and_mirroring(self):
        self.assertEqual(mirror_hand('right'), HAND_LEFT)
        self.assertEqual(mirror_hand('Left'), HAND_RIGHT)
        self.assertEqual(mirror_hand(None), UNSPECIFIED)

        metadata = SampleMetadata(label='a', modality='static', dominant_hand='Direita')
        self.assertEqual(metadata.label, 'A')
        self.assertEqual(metadata.dominant_hand, HAND_RIGHT)

    def test_mirrored_copy_flips_hands_and_records_source(self):
        metadata = SampleMetadata(
            label='A',
            modality='static',
            dominant_hand='right',
            capture_hand='right',
        )

        mirrored = metadata.mirrored_copy('sample_000.npy')

        self.assertTrue(mirrored.mirrored)
        self.assertEqual(mirrored.capture_hand, HAND_LEFT)
        self.assertEqual(mirrored.dominant_hand, HAND_LEFT)
        self.assertEqual(mirrored.source_sample, 'sample_000.npy')
        # A amostra original não é alterada.
        self.assertFalse(metadata.mirrored)
        self.assertEqual(metadata.capture_hand, HAND_RIGHT)

    def test_coverage_counts_annotated_samples_and_lists_subjects(self):
        _write_sample(
            os.path.join(self.temp_dir, 'A'),
            'sample_000.npy',
            SampleMetadata(label='A', modality='static', subject_id='pessoa_01',
                           environment='sala', camera_id='cam_a'),
        )
        _write_sample(
            os.path.join(self.temp_dir, 'A'),
            'sample_001.npy',
            SampleMetadata(label='A', modality='static', subject_id='pessoa_02',
                           environment='escritorio', camera_id='cam_b'),
        )
        _write_sample(os.path.join(self.temp_dir, 'B'), 'sample_000.npy')

        coverage = metadata_coverage(self.temp_dir)

        self.assertEqual(coverage['total_samples'], 3)
        self.assertEqual(coverage['annotated_samples'], 2)
        self.assertAlmostEqual(coverage['coverage'], 2 / 3)
        self.assertEqual(coverage['subjects'], ['pessoa_01', 'pessoa_02'])
        self.assertEqual(coverage['environments'], ['escritorio', 'sala'])
        self.assertEqual(coverage['per_label']['B'], {'total': 1, 'annotated': 0})

        self.assertEqual(len(collect_dataset_metadata(self.temp_dir)), 2)

    def test_coverage_on_missing_directory_is_empty(self):
        coverage = metadata_coverage(os.path.join(self.temp_dir, 'inexistente'))

        self.assertEqual(coverage['total_samples'], 0)
        self.assertEqual(coverage['coverage'], 0.0)


if __name__ == '__main__':
    unittest.main()
