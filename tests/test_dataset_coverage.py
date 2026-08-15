import os
import shutil
import tempfile
import unittest

import numpy as np

from src.dataset.coverage import (
    coverage_for,
    count_label_samples,
    count_label_sources,
    dataset_gaps,
    pending_labels,
    resolve_labels_to_collect,
)
from src.dataset.sample_metadata import SampleMetadata, write_metadata


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dataset_dir, ignore_errors=True)

    def _write_sample(self, label, index, source_dataset=None, mirrored=False):
        label_dir = os.path.join(self.dataset_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        suffix = '_mirror' if mirrored else ''
        path = os.path.join(label_dir, f'sample_{index:03d}{suffix}.npy')
        np.save(path, np.zeros((21, 3), dtype=np.float32))
        if source_dataset is not None:
            write_metadata(
                path,
                SampleMetadata(label=label, modality='static', source_dataset=source_dataset),
            )
        return path

    def test_mirror_samples_do_not_count_towards_target(self):
        self._write_sample('A', 0)
        self._write_sample('A', 0, mirrored=True)
        self.assertEqual(count_label_samples(self.dataset_dir, 'A'), 1)

    def test_missing_label_counts_zero(self):
        self.assertEqual(count_label_samples(self.dataset_dir, 'B'), 0)

    def test_sources_default_to_local_collection(self):
        self._write_sample('A', 0)
        self._write_sample('A', 1, source_dataset='v-librasil')
        self.assertEqual(
            count_label_sources(self.dataset_dir, 'A'),
            {'coleta_local': 1, 'v-librasil': 1},
        )

    def test_pending_labels_lists_only_incomplete_classes(self):
        for index in range(3):
            self._write_sample('A', index)
        self._write_sample('B', 0)

        self.assertEqual(pending_labels(self.dataset_dir, ['A', 'B', 'C'], target=3), ['B', 'C'])

    def test_coverage_reports_missing_count(self):
        self._write_sample('A', 0)
        [item] = coverage_for(self.dataset_dir, ['A'], target=5)
        self.assertEqual(item.missing, 4)
        self.assertFalse(item.complete)

    def test_single_source_class_is_flagged(self):
        self._write_sample('A', 0, source_dataset='v-librasil')
        gaps = dataset_gaps(self.dataset_dir, ['A'], target=1)
        self.assertEqual(gaps['single_source_labels'], ['A'])
        self.assertEqual(gaps['complete_labels'], ['A'])
        self.assertEqual(gaps['pending_labels'], {})

    def test_resolve_skips_complete_labels_by_default(self):
        self._write_sample('A', 0)
        self._write_sample('A', 1)
        labels = resolve_labels_to_collect(
            self.dataset_dir, ['A', 'B'], target=2, only_missing=True, printer=lambda _: None
        )
        self.assertEqual(labels, ['B'])

    def test_resolve_keeps_every_label_when_forced(self):
        self._write_sample('A', 0)
        self._write_sample('A', 1)
        labels = resolve_labels_to_collect(
            self.dataset_dir, ['A', 'B'], target=2, only_missing=False, printer=lambda _: None
        )
        self.assertEqual(labels, ['A', 'B'])


if __name__ == '__main__':
    unittest.main()
