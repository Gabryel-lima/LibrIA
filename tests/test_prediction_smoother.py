import unittest

import numpy as np

from src.inference.prediction_smoother import DuplicateSuppressor, ProbabilitySmoother


class ProbabilitySmootherTests(unittest.TestCase):
    def test_single_update_returns_the_distribution_itself(self):
        smoother = ProbabilitySmoother(window_size=3)

        averaged = smoother.update([0.1, 0.9])

        self.assertTrue(np.allclose(averaged, [0.1, 0.9]))
        self.assertEqual(smoother.best(), (1, 0.9))

    def test_isolated_wrong_prediction_does_not_flip_the_result(self):
        smoother = ProbabilitySmoother(window_size=5)

        for _ in range(4):
            smoother.update([0.9, 0.1])
        smoother.update([0.0, 1.0])  # erro isolado

        index, confidence = smoother.best()
        self.assertEqual(index, 0)
        self.assertAlmostEqual(confidence, (0.9 * 4 + 0.0) / 5)

    def test_window_forgets_older_distributions(self):
        smoother = ProbabilitySmoother(window_size=2)

        smoother.update([1.0, 0.0])
        smoother.update([0.0, 1.0])
        smoother.update([0.0, 1.0])

        self.assertEqual(len(smoother), 2)
        self.assertEqual(smoother.best(), (1, 1.0))

    def test_empty_smoother_has_no_best(self):
        smoother = ProbabilitySmoother()

        self.assertIsNone(smoother.best())
        self.assertIsNone(smoother.average())

    def test_reset_clears_the_window(self):
        smoother = ProbabilitySmoother(window_size=3)
        smoother.update([0.2, 0.8])
        smoother.reset()

        self.assertEqual(len(smoother), 0)
        self.assertIsNone(smoother.best())

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            ProbabilitySmoother(window_size=0)

        smoother = ProbabilitySmoother()
        with self.assertRaises(ValueError):
            smoother.update([])

        smoother.update([0.5, 0.5])
        with self.assertRaises(ValueError):
            smoother.update([0.3, 0.3, 0.4])


class DuplicateSuppressorTests(unittest.TestCase):
    def test_first_emission_always_passes(self):
        suppressor = DuplicateSuppressor(window_seconds=1.0)
        self.assertTrue(suppressor.accept('OI', 0.0))

    def test_same_label_within_the_window_is_suppressed(self):
        suppressor = DuplicateSuppressor(window_seconds=1.0)

        self.assertTrue(suppressor.accept('OI', 0.0))
        self.assertFalse(suppressor.accept('OI', 0.3))
        self.assertFalse(suppressor.accept('OI', 0.6))

    def test_sustained_sign_only_repeats_after_it_stops(self):
        suppressor = DuplicateSuppressor(window_seconds=1.0)

        suppressor.accept('OI', 0.0)
        # Sinal sustentado: cada tentativa renova a janela.
        for timestamp in (0.5, 1.0, 1.5):
            self.assertFalse(suppressor.accept('OI', timestamp))

        # Só depois de 1s sem o sinal ele pode ser emitido de novo.
        self.assertTrue(suppressor.accept('OI', 2.5))

    def test_different_label_is_always_accepted(self):
        suppressor = DuplicateSuppressor(window_seconds=5.0)

        self.assertTrue(suppressor.accept('OI', 0.0))
        self.assertTrue(suppressor.accept('SIM', 0.1))
        self.assertTrue(suppressor.accept('OI', 0.2))

    def test_should_emit_does_not_consume_the_window(self):
        suppressor = DuplicateSuppressor(window_seconds=1.0)
        suppressor.accept('OI', 0.0)

        self.assertFalse(suppressor.should_emit('OI', 0.5))
        self.assertFalse(suppressor.should_emit('OI', 0.5))
        self.assertTrue(suppressor.should_emit('OI', 1.5))

    def test_reset_forgets_the_last_emission(self):
        suppressor = DuplicateSuppressor(window_seconds=10.0)
        suppressor.accept('OI', 0.0)
        suppressor.reset()

        self.assertTrue(suppressor.accept('OI', 0.1))

    def test_negative_window_is_rejected(self):
        with self.assertRaises(ValueError):
            DuplicateSuppressor(window_seconds=-1.0)


if __name__ == '__main__':
    unittest.main()
