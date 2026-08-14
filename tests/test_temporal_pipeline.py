import unittest

import numpy as np

from config.vocabulary import UNKNOWN_LABEL
from src.inference.sign_token import (
    SOURCE_STATIC,
    SOURCE_TEMPORAL,
    STATE_FINAL,
    STATE_PARTIAL,
    STATE_REJECTED,
)
from src.inference.temporal_pipeline import TemporalPipeline

FEATURE_SIZE = 9
SEQUENCE_LENGTH = 4

# Deslocamento por quadro que gera energia bem acima do limiar de início.
MOVING_STEP = 0.1


def _pipeline(temporal_probabilities, static_predictor=None, **overrides):
    """Pipeline determinístico: sem suavização de movimento, limiares fixos."""
    config = {
        'motion_smoothing': 0.0,
        'motion_start_threshold': 0.05,
        'motion_end_threshold': 0.02,
        'min_start_frames': 2,
        'min_end_frames': 2,
        'min_segment_frames': 3,
        'min_duration_seconds': 0.0,
        'max_duration_seconds': 10.0,
        'max_absent_frames': 3,
        'smoothing_window': 5,
        'duplicate_window_seconds': 1.0,
        'emit_partial_tokens': False,
        'partial_interval_frames': 2,
        'temporal_confidence_threshold': 0.7,
        'static_confidence_threshold': 0.75,
        'static_interval_frames': 1,
    }
    config.update(overrides)

    calls = []

    def predictor(sequence):
        calls.append(np.asarray(sequence))
        return temporal_probabilities

    pipeline = TemporalPipeline(
        temporal_predictor=predictor,
        label_map={0: 'OI', 1: 'SIM'},
        sequence_length=SEQUENCE_LENGTH,
        static_predictor=static_predictor,
        config=config,
    )
    return pipeline, calls


def _run(pipeline, moving_counts, still_counts, start_time=0.0, step=0.1):
    """Alterna blocos de movimento e de imobilidade, devolvendo os tokens."""
    tokens = []
    position = 0.0
    timestamp = start_time

    for moving, still in zip(moving_counts, still_counts):
        for _ in range(moving):
            position += MOVING_STEP
            token = pipeline.process_frame(
                np.full(FEATURE_SIZE, position, dtype=np.float32), timestamp
            )
            if token is not None:
                tokens.append(token)
            timestamp += step

        for _ in range(still):
            token = pipeline.process_frame(
                np.full(FEATURE_SIZE, position, dtype=np.float32), timestamp
            )
            if token is not None:
                tokens.append(token)
            timestamp += step

    return tokens


class TemporalPipelineSegmentationTests(unittest.TestCase):
    def test_still_hand_produces_no_temporal_token(self):
        pipeline, calls = _pipeline([0.9, 0.1])

        tokens = _run(pipeline, [0], [20])

        self.assertEqual(tokens, [])
        self.assertEqual(calls, [], 'o modelo temporal não deve ser chamado sem sinal')

    def test_one_sign_produces_one_final_token(self):
        pipeline, calls = _pipeline([0.9, 0.1])

        tokens = _run(pipeline, [6], [4])

        self.assertEqual(len(tokens), 1)
        token = tokens[0]
        self.assertEqual(token.label, 'OI')
        self.assertEqual(token.state, STATE_FINAL)
        self.assertEqual(token.source, SOURCE_TEMPORAL)
        self.assertAlmostEqual(token.confidence, 0.9)
        self.assertGreater(token.duration_seconds, 0.0)
        self.assertEqual(token.sign_type, 'lexical')
        self.assertEqual(len(calls), 1)

    def test_model_receives_the_sequence_length_it_expects(self):
        pipeline, calls = _pipeline([0.9, 0.1])

        _run(pipeline, [9], [3])

        self.assertEqual(calls[0].shape, (SEQUENCE_LENGTH, FEATURE_SIZE))

    def test_two_signs_produce_two_tokens_with_disjoint_windows(self):
        # Janela de deduplicação curta: aqui interessa a segmentação, não o dedup.
        pipeline, _ = _pipeline([0.9, 0.1], duplicate_window_seconds=0.1)

        tokens = _run(pipeline, [5, 5], [4, 4])

        self.assertEqual(len(tokens), 2)
        self.assertLessEqual(tokens[0].end_time, tokens[1].start_time)

    def test_low_confidence_sign_is_rejected_instead_of_translated(self):
        pipeline, _ = _pipeline([0.55, 0.45])

        tokens = _run(pipeline, [6], [4])

        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].state, STATE_REJECTED)
        self.assertEqual(tokens[0].label, UNKNOWN_LABEL)
        self.assertEqual(tokens[0].token, '')

    def test_noise_shorter_than_the_minimum_never_reaches_the_model(self):
        pipeline, calls = _pipeline([0.9, 0.1], min_segment_frames=20)

        tokens = _run(pipeline, [5], [4])

        self.assertEqual(tokens, [])
        self.assertEqual(calls, [])


class TemporalPipelineDuplicateTests(unittest.TestCase):
    def test_repeated_sign_within_the_window_is_emitted_once(self):
        pipeline, _ = _pipeline([0.9, 0.1], duplicate_window_seconds=5.0)

        tokens = _run(pipeline, [5, 5], [4, 4])

        self.assertEqual(len(tokens), 1)

    def test_repeated_sign_after_the_window_is_emitted_again(self):
        pipeline, _ = _pipeline([0.9, 0.1], duplicate_window_seconds=0.1)

        tokens = _run(pipeline, [5, 5], [4, 4])

        self.assertEqual(len(tokens), 2)
        self.assertEqual([token.label for token in tokens], ['OI', 'OI'])


class TemporalPipelinePartialTests(unittest.TestCase):
    def test_partial_tokens_are_emitted_during_the_sign(self):
        pipeline, _ = _pipeline([0.9, 0.1], emit_partial_tokens=True, partial_interval_frames=2)

        tokens = _run(pipeline, [10], [4])

        partials = [token for token in tokens if token.state == STATE_PARTIAL]
        finals = [token for token in tokens if token.state == STATE_FINAL]

        self.assertTrue(partials)
        self.assertEqual(len(finals), 1)
        self.assertTrue(all(token.source == SOURCE_TEMPORAL for token in partials))
        # A parcial cobre do início do sinal até o instante atual.
        self.assertLessEqual(partials[0].start_time, partials[0].end_time)

    def test_partial_tokens_can_be_disabled(self):
        pipeline, _ = _pipeline([0.9, 0.1], emit_partial_tokens=False)

        tokens = _run(pipeline, [10], [4])

        self.assertTrue(all(token.state != STATE_PARTIAL for token in tokens))

    def test_smoothing_makes_the_final_decision_use_the_whole_sign(self):
        # O modelo alterna entre duas classes; a média da janela decide.
        outputs = [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]]
        sequence = iter(outputs * 10)

        pipeline, _ = _pipeline([0.0, 0.0], emit_partial_tokens=True, partial_interval_frames=1)
        pipeline.temporal_predictor = lambda _: next(sequence)

        tokens = _run(pipeline, [10], [4])
        finals = [token for token in tokens if token.state == STATE_FINAL]

        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0].label, 'OI')


class TemporalPipelineStaticFallbackTests(unittest.TestCase):
    def test_static_model_answers_while_the_hand_is_still(self):
        pipeline, calls = _pipeline([0.9, 0.1], static_predictor=lambda _: ('A', 0.95))

        tokens = _run(pipeline, [0], [10])

        self.assertTrue(tokens)
        self.assertEqual(tokens[0].label, 'A')
        self.assertEqual(tokens[0].source, SOURCE_STATIC)
        self.assertEqual(tokens[0].sign_type, 'alphabet')
        self.assertEqual(calls, [], 'o modelo temporal não é consultado sem movimento')

    def test_held_letter_is_not_repeated(self):
        pipeline, _ = _pipeline(
            [0.9, 0.1],
            static_predictor=lambda _: ('A', 0.95),
            duplicate_window_seconds=5.0,
        )

        tokens = _run(pipeline, [0], [30])

        self.assertEqual(len(tokens), 1)

    def test_hand_returning_to_rest_after_a_sign_is_not_read_as_a_letter(self):
        pipeline, _ = _pipeline(
            [0.9, 0.1],
            static_predictor=lambda _: ('A', 0.95),
            static_cooldown_seconds=1.0,
        )

        tokens = _run(pipeline, [6], [6])

        self.assertEqual([token.source for token in tokens], [SOURCE_TEMPORAL])

    def test_static_model_speaks_again_after_the_cooldown(self):
        pipeline, _ = _pipeline(
            [0.9, 0.1],
            static_predictor=lambda _: ('A', 0.95),
            static_cooldown_seconds=0.1,
        )

        tokens = _run(pipeline, [6], [6])
        sources = [token.source for token in tokens]

        self.assertEqual(sources[0], SOURCE_TEMPORAL)
        self.assertIn(SOURCE_STATIC, sources)

    def test_low_confidence_static_prediction_is_dropped(self):
        pipeline, _ = _pipeline([0.9, 0.1], static_predictor=lambda _: ('A', 0.3))

        tokens = _run(pipeline, [0], [10])

        self.assertEqual(tokens, [])

    def test_static_model_is_silent_while_a_sign_is_in_progress(self):
        static_calls = []

        def static_predictor(features):
            static_calls.append(features)
            return 'A', 0.95

        pipeline, _ = _pipeline([0.9, 0.1], static_predictor=static_predictor)

        _run(pipeline, [8], [0])

        # Durante o sinal ativo o fallback estático não é consultado.
        self.assertLessEqual(len(static_calls), 2)


class TemporalPipelineStateTests(unittest.TestCase):
    def test_is_signing_tracks_the_segmenter(self):
        pipeline, _ = _pipeline([0.9, 0.1])

        self.assertFalse(pipeline.is_signing)
        _run(pipeline, [5], [0])
        self.assertTrue(pipeline.is_signing)
        _run(pipeline, [0], [4])
        self.assertFalse(pipeline.is_signing)

    def test_absent_hand_does_not_break_the_pipeline(self):
        pipeline, _ = _pipeline([0.9, 0.1], static_predictor=lambda _: ('A', 0.95))

        for index in range(10):
            token = pipeline.process_frame(None, index * 0.1, hand_present=False)
            self.assertIsNone(token)

        self.assertFalse(pipeline.is_signing)

    def test_reset_clears_every_component(self):
        pipeline, _ = _pipeline([0.9, 0.1])
        _run(pipeline, [5], [0])

        pipeline.reset()

        self.assertFalse(pipeline.is_signing)
        self.assertEqual(pipeline.frame_index, 0)
        self.assertEqual(pipeline.last_energy, 0.0)
        self.assertIsNone(pipeline.last_token)
        self.assertEqual(len(pipeline.buffer), 0)

    def test_last_token_keeps_the_most_recent_emission(self):
        pipeline, _ = _pipeline([0.9, 0.1])

        tokens = _run(pipeline, [6], [4])

        self.assertIs(pipeline.last_token, tokens[-1])

    def test_unmapped_label_index_falls_back_to_the_index(self):
        pipeline, _ = _pipeline([0.1, 0.1, 0.8])

        tokens = _run(pipeline, [6], [4])

        self.assertEqual(tokens[0].label, '2')
        self.assertEqual(tokens[0].sign_type, 'desconhecido')


if __name__ == '__main__':
    unittest.main()
