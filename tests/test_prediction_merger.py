import unittest

from src.inference.prediction_merger import PredictionEvent, PredictionMerger


class PredictionMergerTests(unittest.TestCase):
    def setUp(self):
        self.merger = PredictionMerger(
            temporal_priority_classes=['J', 'Z'],
            temporal_confidence_threshold=0.7,
            static_confidence_threshold=0.75,
            cooldown_seconds=0.5,
        )

    def test_temporal_priority_suppresses_static_during_lock(self):
        temporal_event = PredictionEvent(
            token='J',
            confidence=0.91,
            source='temporal',
            start_time=10.0,
            end_time=10.6,
            frame_index=42,
        )
        static_event = PredictionEvent(
            token='A',
            confidence=0.95,
            source='static',
            start_time=10.7,
            end_time=10.7,
            frame_index=43,
        )

        emitted_temporal = self.merger.submit(temporal_event)
        emitted_static = self.merger.submit(static_event)

        self.assertEqual(emitted_temporal, temporal_event)
        self.assertIsNone(emitted_static)

    def test_static_event_emits_outside_temporal_lock(self):
        temporal_event = PredictionEvent(
            token='Z',
            confidence=0.88,
            source='temporal',
            start_time=5.0,
            end_time=5.4,
            frame_index=10,
        )
        static_event = PredictionEvent(
            token='B',
            confidence=0.84,
            source='static',
            start_time=6.1,
            end_time=6.1,
            frame_index=20,
        )

        self.merger.submit(temporal_event)
        emitted_static = self.merger.submit(static_event)

        self.assertEqual(emitted_static, static_event)

    def test_temporal_non_priority_token_is_ignored(self):
        temporal_event = PredictionEvent(
            token='A',
            confidence=0.99,
            source='temporal',
            start_time=1.0,
            end_time=1.4,
            frame_index=3,
        )

        emitted_event = self.merger.submit(temporal_event)

        self.assertIsNone(emitted_event)


if __name__ == '__main__':
    unittest.main()