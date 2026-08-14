import unittest

from config.vocabulary import (
    MODALITY_STATIC,
    MODALITY_TEMPORAL,
    SIGN_TYPE_ALPHABET,
    SIGN_TYPE_FUNCTIONAL,
    SIGN_TYPE_LEXICAL,
    UNKNOWN_LABEL,
    get_entry,
    get_labels,
    get_modality,
    get_sign_type,
    is_known_label,
    validate_vocabulary,
)


class VocabularyTests(unittest.TestCase):
    def test_vocabulary_is_consistent(self):
        self.assertEqual(validate_vocabulary(), [])

    def test_alphabet_splits_static_and_temporal_letters(self):
        static_letters = get_labels(sign_type=SIGN_TYPE_ALPHABET, modality=MODALITY_STATIC)
        temporal_letters = get_labels(sign_type=SIGN_TYPE_ALPHABET, modality=MODALITY_TEMPORAL)

        self.assertEqual(temporal_letters, ['J', 'Z'])
        self.assertEqual(len(static_letters), 24)
        self.assertNotIn('J', static_letters)
        self.assertNotIn('Z', static_letters)

    def test_lexical_and_functional_families_are_populated(self):
        lexical = get_labels(sign_type=SIGN_TYPE_LEXICAL)
        functional = get_labels(sign_type=SIGN_TYPE_FUNCTIONAL)

        self.assertTrue(lexical)
        self.assertIn('OI', lexical)
        for label in ('ESPACO', 'PAUSA', 'APAGAR', 'CONFIRMAR'):
            self.assertIn(label, functional)

    def test_families_do_not_overlap(self):
        alphabet = set(get_labels(sign_type=SIGN_TYPE_ALPHABET))
        lexical = set(get_labels(sign_type=SIGN_TYPE_LEXICAL))
        functional = set(get_labels(sign_type=SIGN_TYPE_FUNCTIONAL))

        self.assertEqual(alphabet & lexical, set())
        self.assertEqual(alphabet & functional, set())
        self.assertEqual(lexical & functional, set())

    def test_unknown_label_is_resolvable_but_not_part_of_vocabulary(self):
        self.assertFalse(is_known_label(UNKNOWN_LABEL))
        self.assertNotIn(UNKNOWN_LABEL, get_labels())
        self.assertIsNotNone(get_entry(UNKNOWN_LABEL))

    def test_lookup_normalizes_case_and_rejects_unknown_signs(self):
        self.assertTrue(is_known_label('oi'))
        self.assertEqual(get_sign_type('oi'), SIGN_TYPE_LEXICAL)
        self.assertEqual(get_modality('a'), MODALITY_STATIC)
        self.assertEqual(get_modality('j'), MODALITY_TEMPORAL)

        self.assertFalse(is_known_label('SINAL_INEXISTENTE'))
        self.assertIsNone(get_entry('SINAL_INEXISTENTE'))
        self.assertIsNone(get_sign_type('SINAL_INEXISTENTE'))

    def test_invalid_filters_are_rejected(self):
        with self.assertRaises(ValueError):
            get_labels(sign_type='inexistente')
        with self.assertRaises(ValueError):
            get_labels(modality='inexistente')


if __name__ == '__main__':
    unittest.main()
