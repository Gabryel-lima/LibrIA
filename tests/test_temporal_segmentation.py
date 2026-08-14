import unittest

import numpy as np

from src.inference.motion_detector import MotionDetector
from src.inference.sign_segmenter import (
    REASON_HAND_LOST,
    REASON_MAX_DURATION,
    REASON_MOTION_STOPPED,
    SignSegmenter,
)
from src.inference.temporal_buffer import TemporalBuffer, resample_sequence


class TemporalBufferTests(unittest.TestCase):
    def test_buffer_keeps_only_the_last_frames(self):
        buffer = TemporalBuffer(capacity=3)
        for index in range(5):
            buffer.append(np.full(6, index, dtype=np.float32), timestamp=float(index))

        self.assertEqual(len(buffer), 3)
        self.assertTrue(buffer.is_full)
        window = buffer.window()
        self.assertEqual(window.shape, (3, 6))
        self.assertTrue(np.allclose(window[0], 2.0))
        self.assertEqual(buffer.window_span(), (2.0, 4.0))

    def test_window_is_none_until_there_are_enough_frames(self):
        buffer = TemporalBuffer(capacity=4)
        buffer.append(np.zeros(6, dtype=np.float32), 0.0)

        self.assertIsNone(buffer.window())
        self.assertIsNone(buffer.window_span())
        self.assertFalse(buffer.is_full)

    def test_reset_clears_history(self):
        buffer = TemporalBuffer(capacity=2)
        buffer.append(np.zeros(3, dtype=np.float32), 0.0)
        buffer.reset()

        self.assertEqual(len(buffer), 0)

    def test_invalid_capacity_is_rejected(self):
        with self.assertRaises(ValueError):
            TemporalBuffer(capacity=0)


class ResampleTests(unittest.TestCase):
    def test_upsample_and_downsample_hit_the_target_length(self):
        frames = np.arange(12, dtype=np.float32).reshape(4, 3)

        self.assertEqual(resample_sequence(frames, 8).shape, (8, 3))
        self.assertEqual(resample_sequence(frames, 2).shape, (2, 3))

    def test_endpoints_are_preserved(self):
        frames = np.arange(12, dtype=np.float32).reshape(4, 3)

        resampled = resample_sequence(frames, 7)

        self.assertTrue(np.allclose(resampled[0], frames[0]))
        self.assertTrue(np.allclose(resampled[-1], frames[-1]))

    def test_same_length_returns_a_copy(self):
        frames = np.arange(6, dtype=np.float32).reshape(2, 3)

        resampled = resample_sequence(frames, 2)
        resampled[0, 0] = 99.0

        self.assertTrue(np.allclose(resampled, np.array([[99, 1, 2], [3, 4, 5]])))
        self.assertEqual(frames[0, 0], 0.0)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            resample_sequence(np.zeros((0, 3), dtype=np.float32), 4)
        with self.assertRaises(ValueError):
            resample_sequence(np.zeros((4, 3), dtype=np.float32), 0)


class MotionDetectorTests(unittest.TestCase):
    def test_first_frame_has_no_motion(self):
        detector = MotionDetector(smoothing=0.0)
        self.assertEqual(detector.update(np.zeros(9, dtype=np.float32)), 0.0)

    def test_still_hand_keeps_energy_at_zero(self):
        detector = MotionDetector(smoothing=0.0)
        features = np.full(9, 0.5, dtype=np.float32)

        detector.update(features)
        self.assertEqual(detector.update(features), 0.0)

    def test_energy_is_the_mean_euclidean_displacement_per_point(self):
        detector = MotionDetector(smoothing=0.0)
        detector.update(np.zeros(9, dtype=np.float32))

        energy = detector.update(np.full(9, 0.1, dtype=np.float32))

        # Cada ponto se deslocou (0.1, 0.1, 0.1) => norma = 0.1 * sqrt(3).
        self.assertAlmostEqual(energy, 0.1 * np.sqrt(3), places=5)

    def test_smoothing_dampens_isolated_spikes(self):
        detector = MotionDetector(smoothing=0.8)
        detector.update(np.zeros(9, dtype=np.float32))

        spike = detector.update(np.full(9, 1.0, dtype=np.float32))
        raw = np.sqrt(3)

        self.assertLess(spike, raw)
        self.assertGreater(spike, 0.0)

    def test_absent_hand_decays_energy_instead_of_spiking(self):
        detector = MotionDetector(smoothing=0.5)
        detector.update(np.zeros(9, dtype=np.float32))
        detector.update(np.full(9, 0.5, dtype=np.float32))
        energy_before = detector.energy

        decayed = detector.update(None)
        self.assertLess(decayed, energy_before)

        # Ao reaparecer longe, não há pico artificial: o histórico foi zerado.
        self.assertLess(detector.update(np.full(9, 5.0, dtype=np.float32)), decayed + 1e-6)

    def test_has_reading_requires_two_consecutive_frames(self):
        detector = MotionDetector(smoothing=0.0)

        self.assertFalse(detector.has_reading)
        detector.update(np.zeros(9, dtype=np.float32))
        self.assertFalse(detector.has_reading, 'um quadro isolado não mede movimento')

        detector.update(np.zeros(9, dtype=np.float32))
        self.assertTrue(detector.has_reading)

        detector.update(None)
        self.assertFalse(detector.has_reading, 'perder a mão invalida a leitura')

    def test_reset_clears_state(self):
        detector = MotionDetector(smoothing=0.0)
        detector.update(np.zeros(9, dtype=np.float32))
        detector.update(np.full(9, 1.0, dtype=np.float32))
        detector.reset()

        self.assertEqual(detector.energy, 0.0)
        self.assertEqual(detector.update(np.full(9, 1.0, dtype=np.float32)), 0.0)

    def test_invalid_smoothing_is_rejected(self):
        with self.assertRaises(ValueError):
            MotionDetector(smoothing=1.0)


class SignSegmenterTests(unittest.TestCase):
    def _segmenter(self, **overrides):
        params = dict(
            start_threshold=0.05,
            end_threshold=0.02,
            min_start_frames=2,
            min_end_frames=2,
            min_frames=3,
            min_duration_seconds=0.0,
            max_duration_seconds=10.0,
            max_absent_frames=3,
        )
        params.update(overrides)
        return SignSegmenter(**params)

    def _feed(self, segmenter, energies, start_time=0.0, step=0.1, hand_present=True):
        """Alimenta o segmentador e devolve os segmentos emitidos."""
        segments = []
        for index, energy in enumerate(energies):
            timestamp = start_time + index * step
            features = None if not hand_present else np.full(9, index * 0.01, dtype=np.float32)
            segment = segmenter.update(features, timestamp, energy, hand_present=hand_present)
            if segment is not None:
                segments.append(segment)
        return segments

    def test_hysteresis_requires_more_energy_to_start_than_to_stop(self):
        with self.assertRaises(ValueError):
            SignSegmenter(start_threshold=0.01, end_threshold=0.05)

    def test_still_hand_never_starts_a_segment(self):
        segmenter = self._segmenter()

        segments = self._feed(segmenter, [0.0] * 20)

        self.assertEqual(segments, [])
        self.assertFalse(segmenter.is_active)

    def test_movement_then_stillness_produces_one_segment(self):
        segmenter = self._segmenter()

        segments = self._feed(segmenter, [0.1] * 6 + [0.0] * 4)

        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertEqual(segment.reason, REASON_MOTION_STOPPED)
        self.assertFalse(segmenter.is_active)
        self.assertGreater(segment.duration_seconds, 0.0)

    def test_trailing_still_frames_are_trimmed_from_the_segment(self):
        segmenter = self._segmenter(min_end_frames=2)

        segments = self._feed(segmenter, [0.1] * 6 + [0.0] * 2)

        # 6 quadros em movimento + 2 parados; os 2 parados saem do segmento.
        self.assertEqual(segments[0].frame_count, 6)

    def test_brief_pause_does_not_split_a_single_sign(self):
        segmenter = self._segmenter(min_end_frames=3)

        # Uma pausa de 2 quadros no meio não atinge min_end_frames.
        segments = self._feed(segmenter, [0.1] * 4 + [0.0] * 2 + [0.1] * 4 + [0.0] * 3)

        self.assertEqual(len(segments), 1)
        self.assertGreaterEqual(segments[0].frame_count, 8)

    def test_intermediate_energy_keeps_the_sign_active(self):
        segmenter = self._segmenter()

        # 0.03 está entre end_threshold e start_threshold: nem começa, nem encerra.
        segments = self._feed(segmenter, [0.1] * 4 + [0.03] * 6 + [0.0] * 2)

        self.assertEqual(len(segments), 1)

    def test_segments_shorter_than_the_minimum_are_discarded_as_noise(self):
        segmenter = self._segmenter(min_frames=10)

        segments = self._feed(segmenter, [0.1] * 4 + [0.0] * 3)

        self.assertEqual(segments, [])
        self.assertFalse(segmenter.is_active)

    def test_segment_below_minimum_duration_is_discarded(self):
        segmenter = self._segmenter(min_duration_seconds=1.0)

        segments = self._feed(segmenter, [0.1] * 5 + [0.0] * 3, step=0.01)

        self.assertEqual(segments, [])

    def test_sign_is_closed_when_it_exceeds_the_maximum_duration(self):
        segmenter = self._segmenter(max_duration_seconds=0.5)

        segments = self._feed(segmenter, [0.1] * 20, step=0.1)

        self.assertTrue(segments)
        self.assertEqual(segments[0].reason, REASON_MAX_DURATION)

    def test_losing_the_hand_closes_the_sign(self):
        segmenter = self._segmenter(max_absent_frames=2)

        for index, energy in enumerate([0.1] * 5):
            segmenter.update(np.full(9, index * 0.01, dtype=np.float32), index * 0.1, energy)

        self.assertTrue(segmenter.is_active)

        segment = None
        for index in range(2):
            segment = segmenter.update(None, 0.5 + index * 0.1, 0.1, hand_present=False) or segment

        self.assertIsNotNone(segment)
        self.assertEqual(segment.reason, REASON_HAND_LOST)
        self.assertFalse(segmenter.is_active)

    def test_two_signs_separated_by_stillness_produce_two_segments(self):
        segmenter = self._segmenter()

        segments = self._feed(segmenter, [0.1] * 5 + [0.0] * 4 + [0.1] * 5 + [0.0] * 4)

        self.assertEqual(len(segments), 2)
        self.assertLess(segments[0].end_time, segments[1].start_time)

    def test_the_first_frames_of_the_sign_are_kept(self):
        segmenter = self._segmenter(min_start_frames=3)

        segments = self._feed(segmenter, [0.1] * 6 + [0.0] * 3)

        # Os 3 quadros gastos confirmando o início entram no segmento.
        self.assertEqual(segments[0].frame_count, 6)
        self.assertAlmostEqual(segments[0].start_time, 0.0)

    def test_reset_drops_an_in_progress_sign(self):
        segmenter = self._segmenter()
        self._feed(segmenter, [0.1] * 4)
        self.assertTrue(segmenter.is_active)

        segmenter.reset()

        self.assertFalse(segmenter.is_active)
        self.assertEqual(segmenter.frame_count, 0)


if __name__ == '__main__':
    unittest.main()
