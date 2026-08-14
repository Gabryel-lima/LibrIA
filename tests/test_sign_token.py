import unittest

from config.vocabulary import UNKNOWN_LABEL
from src.inference.sign_token import (
    SOURCE_STATIC,
    SOURCE_TEMPORAL,
    STATE_FINAL,
    STATE_PARTIAL,
    STATE_REJECTED,
    build_rejected_token,
    build_token,
)


class SignTokenTests(unittest.TestCase):
    def test_sign_type_is_resolved_from_vocabulary(self):
        letter = build_token('A', 0.9, 1.0, 1.0, SOURCE_STATIC)
        word = build_token('oi', 0.9, 1.0, 2.0, SOURCE_TEMPORAL)
        gesture = build_token('APAGAR', 0.9, 1.0, 2.0, SOURCE_TEMPORAL)

        self.assertEqual(letter.sign_type, 'alphabet')
        self.assertEqual(word.sign_type, 'lexical')
        self.assertEqual(word.label, 'OI')
        self.assertEqual(gesture.sign_type, 'functional')

    def test_unknown_sign_falls_back_to_unspecified_type(self):
        token = build_token('SINAL_INEXISTENTE', 0.9, 1.0, 2.0, SOURCE_TEMPORAL)
        self.assertEqual(token.sign_type, 'desconhecido')

    def test_token_text_uses_translation_and_is_empty_for_gestures(self):
        self.assertEqual(build_token('OI', 0.9, 0.0, 1.0, SOURCE_TEMPORAL).token, 'oi')
        self.assertEqual(build_token('A', 0.9, 0.0, 0.0, SOURCE_STATIC).token, 'A')
        self.assertEqual(build_token('PAUSA', 0.9, 0.0, 1.0, SOURCE_TEMPORAL).token, '')

    def test_duration_and_finalization_state(self):
        token = build_token('OI', 0.9, 10.0, 11.5, SOURCE_TEMPORAL)

        self.assertAlmostEqual(token.duration_seconds, 1.5)
        self.assertTrue(token.is_final)
        self.assertFalse(token.is_rejected)

        partial = build_token('OI', 0.5, 10.0, 10.5, SOURCE_TEMPORAL, state=STATE_PARTIAL)
        self.assertFalse(partial.is_final)
        self.assertEqual(partial.state, STATE_PARTIAL)

    def test_rejected_token_is_unknown_and_has_no_text(self):
        token = build_rejected_token(1.0, 2.0, SOURCE_TEMPORAL, confidence=0.3, frame_count=12)

        self.assertEqual(token.label, UNKNOWN_LABEL)
        self.assertEqual(token.state, STATE_REJECTED)
        self.assertTrue(token.is_rejected)
        self.assertFalse(token.is_final)
        self.assertEqual(token.token, '')
        self.assertEqual(token.frame_count, 12)

    def test_invalid_state_is_rejected(self):
        with self.assertRaises(ValueError):
            build_token('A', 0.9, 0.0, 0.0, SOURCE_STATIC, state='inexistente')

    def test_serialization_carries_the_full_contract(self):
        payload = build_token('OI', 0.87, 1.0, 2.0, SOURCE_TEMPORAL, frame_count=24).to_dict()

        for key in (
            'label', 'token', 'confidence', 'start_time', 'end_time',
            'duration_seconds', 'source', 'state', 'sign_type', 'frame_count',
        ):
            self.assertIn(key, payload)

        self.assertEqual(payload['state'], STATE_FINAL)
        self.assertEqual(payload['sign_type'], 'lexical')


if __name__ == '__main__':
    unittest.main()
