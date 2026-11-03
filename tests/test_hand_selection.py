"""Seleção de mão com resultados falsos no formato do MediaPipe (sem MediaPipe)."""

import unittest
from types import SimpleNamespace

from config.settings import HAND_SELECTION_CONFIG
from utils.hand_selection import (
    HAND_LEFT,
    HAND_RIGHT,
    STRATEGY_DOMINANT,
    STRATEGY_FIRST,
    STRATEGY_LARGEST,
    STRATEGY_MOST_CONFIDENT,
    HandSelector,
    choose_candidate,
    hand_candidates,
)


def _hand(wrist_x, wrist_y, size):
    """21 pontos num quadrado de lado ``size`` com o punho em (wrist_x, wrist_y)."""
    points = [SimpleNamespace(x=wrist_x, y=wrist_y, z=0.0)]
    for i in range(1, 21):
        offset = size * (i / 20.0)
        points.append(SimpleNamespace(x=wrist_x + offset, y=wrist_y - offset, z=0.0))
    return SimpleNamespace(landmark=points)


def _handedness(label, score):
    return SimpleNamespace(classification=[SimpleNamespace(label=label, score=score)])


def _results(*hands):
    """Cada mão: (wrist_x, wrist_y, size, mediapipe_label, score)."""
    return SimpleNamespace(
        multi_hand_landmarks=[_hand(x, y, size) for x, y, size, _, _ in hands],
        multi_handedness=[_handedness(label, score) for _, _, _, label, score in hands],
    )


class HandCandidatesTest(unittest.TestCase):
    def test_no_hands(self):
        self.assertEqual(hand_candidates(SimpleNamespace(multi_hand_landmarks=None)), [])

    def test_unmirrored_image_swaps_mediapipe_label(self):
        # O MediaPipe supõe selfie espelhada: "Right" num quadro não espelhado
        # é a mão esquerda da pessoa.
        results = _results((0.5, 0.5, 0.1, 'Right', 0.9))
        self.assertEqual(hand_candidates(results)[0].person_hand, HAND_LEFT)
        self.assertEqual(hand_candidates(results, image_is_mirrored=True)[0].person_hand, HAND_RIGHT)

    def test_missing_handedness_is_tolerated(self):
        results = SimpleNamespace(multi_hand_landmarks=[_hand(0.5, 0.5, 0.1)], multi_handedness=None)
        candidate = hand_candidates(results)[0]
        self.assertIsNone(candidate.score)
        self.assertIsNone(candidate.person_hand)

    def test_bbox_area(self):
        candidate = hand_candidates(_results((0.2, 0.8, 0.4, 'Left', 0.9)))[0]
        self.assertAlmostEqual(candidate.bbox_area, 0.16)
        self.assertEqual(candidate.wrist, (0.2, 0.8))


class ChooseCandidateTest(unittest.TestCase):
    def setUp(self):
        # Mão 0: pequena, muito confiante, rótulo MP "Right" (= esquerda real).
        # Mão 1: grande, menos confiante, rótulo MP "Left" (= direita real).
        self.candidates = hand_candidates(
            _results((0.2, 0.5, 0.1, 'Right', 0.99), (0.7, 0.5, 0.3, 'Left', 0.80))
        )

    def test_first_keeps_legacy_behavior(self):
        self.assertEqual(choose_candidate(self.candidates, STRATEGY_FIRST).index, 0)

    def test_largest(self):
        self.assertEqual(choose_candidate(self.candidates, STRATEGY_LARGEST).index, 1)

    def test_most_confident(self):
        self.assertEqual(choose_candidate(self.candidates, STRATEGY_MOST_CONFIDENT).index, 0)

    def test_dominant_uses_real_hand(self):
        self.assertEqual(choose_candidate(self.candidates, STRATEGY_DOMINANT, 'right').index, 1)
        self.assertEqual(choose_candidate(self.candidates, STRATEGY_DOMINANT, 'esquerda').index, 0)

    def test_dominant_falls_back_to_largest_when_side_is_absent(self):
        only_left = [self.candidates[0]]
        self.assertEqual(choose_candidate(only_left, STRATEGY_DOMINANT, 'right').index, 0)

    def test_unknown_strategy(self):
        with self.assertRaises(ValueError):
            choose_candidate(self.candidates, 'random')
        with self.assertRaises(ValueError):
            HandSelector(strategy='random')

    def test_empty(self):
        self.assertIsNone(choose_candidate([], STRATEGY_LARGEST))


class HandSelectorTrackingTest(unittest.TestCase):
    def test_tracking_prevents_switching_between_similar_hands(self):
        selector = HandSelector(strategy=STRATEGY_LARGEST, tracking_max_jump=0.15)
        # Quadro 1: a mão da direita do quadro é um pouco maior.
        first = selector.select(_results((0.2, 0.5, 0.20, 'Right', 0.9), (0.7, 0.5, 0.21, 'Left', 0.9)))
        self.assertAlmostEqual(first.wrist[0], 0.7)
        # Quadro 2: a outra passa a ser maior, e o MediaPipe inverte a ordem.
        # Sem rastreamento a escolha trocaria de mão no meio do sinal.
        second = selector.select(_results((0.72, 0.5, 0.20, 'Left', 0.9), (0.2, 0.5, 0.22, 'Right', 0.9)))
        self.assertAlmostEqual(second.wrist[0], 0.72)
        self.assertEqual(second.index, 0)

    def test_large_jump_falls_back_to_strategy(self):
        selector = HandSelector(strategy=STRATEGY_LARGEST, tracking_max_jump=0.15)
        selector.select(_results((0.7, 0.5, 0.2, 'Left', 0.9)))
        chosen = selector.select(_results((0.1, 0.1, 0.1, 'Left', 0.9), (0.3, 0.9, 0.3, 'Right', 0.9)))
        self.assertAlmostEqual(chosen.wrist[0], 0.3)

    def test_losing_the_hand_resets_tracking(self):
        selector = HandSelector(strategy=STRATEGY_LARGEST, tracking_max_jump=0.15)
        selector.select(_results((0.7, 0.5, 0.1, 'Left', 0.9)))
        self.assertIsNone(selector.select(SimpleNamespace(multi_hand_landmarks=[])))
        chosen = selector.select(_results((0.71, 0.5, 0.1, 'Left', 0.9), (0.2, 0.5, 0.3, 'Right', 0.9)))
        self.assertAlmostEqual(chosen.wrist[0], 0.2)

    def test_tracking_disabled(self):
        selector = HandSelector(strategy=STRATEGY_LARGEST, tracking_max_jump=None)
        selector.select(_results((0.7, 0.5, 0.21, 'Left', 0.9)))
        chosen = selector.select(_results((0.71, 0.5, 0.20, 'Left', 0.9), (0.2, 0.5, 0.22, 'Right', 0.9)))
        self.assertAlmostEqual(chosen.wrist[0], 0.2)

    def test_dominant_hand_entering_replaces_tracked_fallback(self):
        selector = HandSelector(strategy=STRATEGY_DOMINANT, dominant_hand='right')
        # Só a esquerda real (rótulo MP "Right") está visível: fallback.
        self.assertEqual(selector.select(_results((0.3, 0.5, 0.2, 'Right', 0.9))).person_hand, HAND_LEFT)
        # A direita real entra perto: passa a ser a escolhida, apesar do rastreamento.
        chosen = selector.select(_results((0.31, 0.5, 0.2, 'Right', 0.9), (0.35, 0.5, 0.1, 'Left', 0.9)))
        self.assertEqual(chosen.person_hand, HAND_RIGHT)

    def test_first_strategy_ignores_tracking(self):
        selector = HandSelector(strategy=STRATEGY_FIRST)
        selector.select(_results((0.7, 0.5, 0.2, 'Left', 0.9)))
        chosen = selector.select(_results((0.2, 0.5, 0.2, 'Right', 0.9), (0.7, 0.5, 0.2, 'Left', 0.9)))
        self.assertEqual(chosen.index, 0)

    def test_from_config_matches_settings(self):
        selector = HandSelector.from_config(HAND_SELECTION_CONFIG, dominant_hand='left')
        self.assertEqual(selector.strategy, HAND_SELECTION_CONFIG['strategy'])
        self.assertEqual(selector.dominant_hand, HAND_LEFT)
        self.assertFalse(selector.image_is_mirrored)
        self.assertGreaterEqual(HAND_SELECTION_CONFIG['max_num_hands'], 2)



class _FakeHands:
    """Imita ``mp.solutions.hands.Hands``: devolve resultados pré-definidos."""

    def __init__(self, *results):
        self._results = list(results)

    def process(self, _rgb_frame):
        return self._results.pop(0)


class IntegrationTest(unittest.TestCase):
    """Os pontos de entrada usam a mão escolhida, não ``multi_hand_landmarks[0]``."""

    def setUp(self):
        import numpy as np

        self.np = np
        self.frame = np.zeros((8, 8, 3), dtype=np.uint8)
        # Mão 0 pequena (rótulo MP "Left" = direita real), mão 1 grande.
        self.results = _results((0.2, 0.5, 0.05, 'Left', 0.95), (0.6, 0.6, 0.3, 'Right', 0.95))

    def _expected(self, index):
        from config.settings import FEATURE_MODE
        from utils.helpers import extract_landmarks_by_mode

        hand = self.results.multi_hand_landmarks[index]
        return self.np.asarray(extract_landmarks_by_mode(hand.landmark, FEATURE_MODE), dtype=self.np.float32).ravel()

    def test_collect_uses_selected_hand_and_real_handedness(self):
        from scripts import collect_dataset

        candidate = collect_dataset._select_hand_from_image(self.results)
        sample = collect_dataset._extract_valid_sample(candidate)
        expected_index = 1 if HAND_SELECTION_CONFIG['strategy'] == STRATEGY_LARGEST else candidate.index
        self.assertEqual(candidate.index, expected_index)
        self.np.testing.assert_allclose(self.np.asarray(sample).ravel(), self._expected(candidate.index))
        # Rótulo MP "Right" num quadro não espelhado = mão esquerda real.
        if candidate.index == 1:
            self.assertEqual(collect_dataset._handedness_label(candidate), HAND_LEFT)
        self.assertAlmostEqual(collect_dataset._handedness_score(candidate), 0.95)

    def test_collect_rejects_low_confidence_and_missing_hand(self):
        from config.settings import COLLECTION_CONFIG
        from scripts import collect_dataset

        low = _results((0.5, 0.5, 0.2, 'Left', COLLECTION_CONFIG['min_detection_confidence'] / 2))
        self.assertIsNone(collect_dataset._extract_valid_sample(collect_dataset._select_hand_from_image(low)))
        self.assertIsNone(collect_dataset._extract_valid_sample(None))
        self.assertIsNone(collect_dataset._handedness_label(None))

    def test_collect_frame_backfill(self):
        from scripts import collect_dataset

        sample = collect_dataset._extract_valid_sample_from_frame(self.frame, _FakeHands(self.results))
        expected = collect_dataset._select_hand_from_image(self.results).index
        self.np.testing.assert_allclose(self.np.asarray(sample).ravel(), self._expected(expected))

    def test_video_ingest_frame_sample(self):
        from src.dataset import video_ingest

        sample = video_ingest.extract_frame_sample(self.frame, _FakeHands(self.results))
        expected = choose_candidate(
            hand_candidates(self.results), HAND_SELECTION_CONFIG['strategy'], HAND_SELECTION_CONFIG['dominant_hand']
        ).index
        self.np.testing.assert_allclose(self.np.asarray(sample).ravel(), self._expected(expected))

        empty = SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)
        self.assertIsNone(video_ingest.extract_frame_sample(self.frame, _FakeHands(empty)))

    def test_video_ingest_follows_same_hand_with_selector(self):
        from src.dataset import video_ingest

        selector = HandSelector(strategy=STRATEGY_LARGEST, tracking_max_jump=0.15)
        first = _results((0.2, 0.5, 0.20, 'Left', 0.9), (0.7, 0.5, 0.21, 'Right', 0.9))
        second = _results((0.71, 0.5, 0.20, 'Right', 0.9), (0.2, 0.5, 0.22, 'Left', 0.9))
        hands = _FakeHands(first, second)
        video_ingest.extract_frame_sample(self.frame, hands, hand_selector=selector)
        sample = video_ingest.extract_frame_sample(self.frame, hands, hand_selector=selector)
        self.results = second
        self.np.testing.assert_allclose(self.np.asarray(sample).ravel(), self._expected(0))


if __name__ == '__main__':
    unittest.main()
