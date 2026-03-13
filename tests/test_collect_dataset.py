import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from scripts.collect_dataset import _backfill_static_samples_from_frames


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


if __name__ == '__main__':
    unittest.main()