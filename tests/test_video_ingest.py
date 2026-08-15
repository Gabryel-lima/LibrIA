import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from config.vocabulary import MODALITY_STATIC, MODALITY_TEMPORAL
from src.dataset.video_ingest import (
    LABEL_FROM_FILENAME,
    IngestState,
    discover_source_items,
    load_label_map,
    normalize_label,
    resample_sequence,
)


class NormalizeLabelTests(unittest.TestCase):
    def test_removes_accents_and_spaces(self):
        self.assertEqual(normalize_label('Tudo bem?'), 'TUDO_BEM')
        self.assertEqual(normalize_label('  não  '), 'NAO')
        self.assertEqual(normalize_label('por-favor'), 'POR_FAVOR')

    def test_keeps_labels_already_normalized(self):
        self.assertEqual(normalize_label('TUDO_BEM'), 'TUDO_BEM')
        self.assertEqual(normalize_label('J'), 'J')


class ResampleSequenceTests(unittest.TestCase):
    def test_resamples_to_fixed_length(self):
        frames = [np.full((21, 3), index, dtype=np.float32) for index in range(7)]
        sequence = resample_sequence(frames, 30)
        self.assertEqual(sequence.shape, (30, 21, 3))

    def test_keeps_first_and_last_frames(self):
        frames = [np.full((21, 3), index, dtype=np.float32) for index in range(10)]
        sequence = resample_sequence(frames, 30)
        self.assertEqual(sequence[0][0][0], 0.0)
        self.assertEqual(sequence[-1][0][0], 9.0)

    def test_upsamples_short_clips(self):
        frames = [np.zeros((21, 3), dtype=np.float32), np.ones((21, 3), dtype=np.float32)]
        self.assertEqual(resample_sequence(frames, 30).shape, (30, 21, 3))

    def test_rejects_empty_sequence(self):
        with self.assertRaises(ValueError):
            resample_sequence([], 30)


class DiscoverSourceItemsTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _touch(self, *parts):
        path = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as file_obj:
            file_obj.write(b'0')
        return path

    def test_label_comes_from_top_level_directory(self):
        self._touch('OI', 'clip1.mp4')
        self._touch('OI', 'clip2.mp4')
        items, _ = discover_source_items(self.root, MODALITY_TEMPORAL)
        self.assertEqual({item.label for item in items}, {'OI'})
        self.assertEqual(len(items), 2)

    def test_skips_terms_outside_vocabulary(self):
        self._touch('OI', 'clip.mp4')
        self._touch('CADEIRA', 'clip.mp4')
        items, skipped = discover_source_items(self.root, MODALITY_TEMPORAL)
        self.assertEqual([item.label for item in items], ['OI'])
        self.assertEqual(skipped, {'CADEIRA': 1})

    def test_label_map_translates_source_terms(self):
        self._touch('tudo bem', 'clip.mp4')
        items, skipped = discover_source_items(
            self.root, MODALITY_TEMPORAL, label_map={'TUDO_BEM': 'TUDO_BEM'}
        )
        self.assertEqual([item.label for item in items], ['TUDO_BEM'])
        self.assertFalse(skipped)

    def test_accepts_unknown_labels_when_vocabulary_is_open(self):
        self._touch('CADEIRA', 'clip.mp4')
        items, skipped = discover_source_items(
            self.root, MODALITY_TEMPORAL, only_vocabulary=False
        )
        self.assertEqual([item.label for item in items], ['CADEIRA'])
        self.assertFalse(skipped)

    def test_temporal_ignores_still_images(self):
        self._touch('OI', 'thumb.png')
        items, _ = discover_source_items(self.root, MODALITY_TEMPORAL)
        self.assertEqual(items, [])

    def test_static_accepts_images(self):
        self._touch('A', 'hand.png')
        items, _ = discover_source_items(self.root, MODALITY_STATIC)
        self.assertEqual([item.label for item in items], ['A'])

    def test_subject_pattern_identifies_signer(self):
        self._touch('OI', 'signer03_take1.mp4')
        items, _ = discover_source_items(
            self.root,
            MODALITY_TEMPORAL,
            subject_pattern=r'signer(?P<subject>\d+)',
            default_subject='base',
        )
        self.assertEqual(items[0].subject_id, '03')

    def test_default_subject_when_pattern_does_not_match(self):
        self._touch('OI', 'clip.mp4')
        items, _ = discover_source_items(
            self.root,
            MODALITY_TEMPORAL,
            subject_pattern=r'signer(?P<subject>\d+)',
            default_subject='base',
        )
        self.assertEqual(items[0].subject_id, 'base')

    def test_label_can_come_from_the_filename(self):
        self._touch('Sinalizador01', 'OI.mp4')
        items, _ = discover_source_items(
            self.root, MODALITY_TEMPORAL, label_from=LABEL_FROM_FILENAME
        )
        self.assertEqual([item.label for item in items], ['OI'])

    def test_label_regex_extracts_sign_from_mixed_names(self):
        self._touch('Sinalizador01', '01_OI_take3.mp4')
        items, _ = discover_source_items(
            self.root, MODALITY_TEMPORAL, label_pattern=r'_(?P<label>[A-Z_]+)_take'
        )
        self.assertEqual([item.label for item in items], ['OI'])

    def test_files_without_label_match_are_reported_not_ingested(self):
        self._touch('Sinalizador01', 'sem_padrao.mp4')
        items, skipped = discover_source_items(
            self.root, MODALITY_TEMPORAL, label_pattern=r'_(?P<label>[A-Z]+)_take'
        )
        self.assertEqual(items, [])
        self.assertEqual(sum(skipped.values()), 1)

    def test_missing_directory_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            discover_source_items(os.path.join(self.root, 'ausente'), MODALITY_TEMPORAL)


class LoadLabelMapTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_normalizes_keys_and_values(self):
        path = os.path.join(self.root, 'map.json')
        with open(path, 'w', encoding='utf-8') as file_obj:
            json.dump({'Tudo bem': 'tudo_bem'}, file_obj)
        self.assertEqual(load_label_map(path), {'TUDO_BEM': 'TUDO_BEM'})

    def test_empty_path_returns_empty_map(self):
        self.assertEqual(load_label_map(None), {})


class IngestStateTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.video = os.path.join(self.root, 'clip.mp4')
        with open(self.video, 'wb') as file_obj:
            file_obj.write(b'0000')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_records_survive_reload(self):
        state = IngestState(self.root)
        self.assertFalse(state.already_ingested(self.video))
        state.record(self.video, 'OI', 'seq_000', 'v-librasil')
        state.save()

        self.assertTrue(IngestState(self.root).already_ingested(self.video))

    def test_changed_file_is_reprocessed(self):
        state = IngestState(self.root)
        state.record(self.video, 'OI', 'seq_000', 'v-librasil')
        state.save()

        with open(self.video, 'wb') as file_obj:
            file_obj.write(b'000000000')

        self.assertFalse(IngestState(self.root).already_ingested(self.video))

    def test_corrupted_state_does_not_block_ingestion(self):
        with open(os.path.join(self.root, '.ingest_state.json'), 'w', encoding='utf-8') as file_obj:
            file_obj.write('{ nao é json')

        self.assertFalse(IngestState(self.root).already_ingested(self.video))


if __name__ == '__main__':
    unittest.main()
