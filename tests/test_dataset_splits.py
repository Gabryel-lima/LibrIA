import unittest

from src.dataset.sample_metadata import SampleMetadata
from src.evaluation.dataset_splits import (
    SPLIT_NAMES,
    TEST,
    TRAIN,
    VALIDATION,
    assert_no_subject_leakage,
    find_subject_leakage,
    split_metadata_by_person,
    split_subjects_by_person,
)


class SplitByPersonTests(unittest.TestCase):
    def _subject_ids(self, counts):
        subject_ids = []
        for subject, count in counts.items():
            subject_ids.extend([subject] * count)
        return subject_ids

    def test_no_subject_appears_in_more_than_one_split(self):
        subject_ids = self._subject_ids({f'p{index:02d}': 10 for index in range(6)})

        split = split_subjects_by_person(subject_ids)

        assert_no_subject_leakage(split)
        self.assertEqual(find_subject_leakage(split), [])

        all_subjects = [
            subject for name in SPLIT_NAMES for subject in split.subjects[name]
        ]
        self.assertEqual(len(all_subjects), len(set(all_subjects)))
        self.assertEqual(set(all_subjects), set(subject_ids))

    def test_every_sample_is_assigned_exactly_once(self):
        subject_ids = self._subject_ids({'ana': 12, 'bruno': 8, 'caio': 5, 'duda': 3})

        split = split_subjects_by_person(subject_ids)

        all_indices = split.train_indices + split.validation_indices + split.test_indices
        self.assertEqual(sorted(all_indices), list(range(len(subject_ids))))

    def test_indices_of_a_split_belong_only_to_its_subjects(self):
        subject_ids = self._subject_ids({'ana': 6, 'bruno': 6, 'caio': 6})

        split = split_subjects_by_person(subject_ids)

        for name in SPLIT_NAMES:
            members = set(split.subjects[name])
            for index in split.indices[name]:
                self.assertIn(subject_ids[index], members)

    def test_split_is_deterministic(self):
        subject_ids = self._subject_ids({'ana': 7, 'bruno': 5, 'caio': 4, 'duda': 9})

        first = split_subjects_by_person(subject_ids)
        second = split_subjects_by_person(subject_ids)

        self.assertEqual(first.subjects, second.subjects)
        self.assertEqual(first.indices, second.indices)

    def test_proportions_stay_close_to_target(self):
        subject_ids = self._subject_ids({f'p{index:02d}': 10 for index in range(10)})

        split = split_subjects_by_person(
            subject_ids, ratios={TRAIN: 0.6, VALIDATION: 0.2, TEST: 0.2}
        )

        total = len(subject_ids)
        self.assertAlmostEqual(len(split.train_indices) / total, 0.6, delta=0.1)
        self.assertAlmostEqual(len(split.validation_indices) / total, 0.2, delta=0.1)
        self.assertAlmostEqual(len(split.test_indices) / total, 0.2, delta=0.1)

    def test_too_few_subjects_fails_loudly(self):
        subject_ids = self._subject_ids({'ana': 20, 'bruno': 20})

        with self.assertRaises(ValueError) as context:
            split_subjects_by_person(subject_ids)

        self.assertIn('pessoas distintas', str(context.exception))

    def test_too_few_subjects_can_be_allowed_explicitly(self):
        subject_ids = self._subject_ids({'ana': 20, 'bruno': 20})

        split = split_subjects_by_person(subject_ids, allow_empty_splits=True)

        assert_no_subject_leakage(split)
        self.assertEqual(len(split.train_indices) + len(split.validation_indices)
                         + len(split.test_indices), len(subject_ids))

    def test_ratios_are_normalized_and_validated(self):
        subject_ids = self._subject_ids({'ana': 4, 'bruno': 4, 'caio': 4})

        split = split_subjects_by_person(subject_ids, ratios={TRAIN: 6, VALIDATION: 2, TEST: 2})
        assert_no_subject_leakage(split)

        with self.assertRaises(ValueError):
            split_subjects_by_person(subject_ids, ratios={'treino': 1.0})
        with self.assertRaises(ValueError):
            split_subjects_by_person(subject_ids, ratios={TRAIN: 0.0, VALIDATION: 0.0, TEST: 0.0})

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            split_subjects_by_person([])

    def test_split_from_metadata_objects(self):
        metadata_list = [
            SampleMetadata(label='A', modality='static', subject_id=subject)
            for subject in ['ana'] * 6 + ['bruno'] * 4 + ['caio'] * 2
        ]

        split = split_metadata_by_person(metadata_list)

        assert_no_subject_leakage(split)
        self.assertEqual(split.split_of_subject('ana'), TRAIN)
        self.assertIsNone(split.split_of_subject('inexistente'))
        self.assertEqual(
            sum(stats['samples'] for stats in split.summary().values()),
            len(metadata_list),
        )


if __name__ == '__main__':
    unittest.main()
