import unittest

import numpy as np

from config.vocabulary import UNKNOWN_LABEL
from src.evaluation.metrics import (
    apply_rejection,
    compute_latency_stats,
    evaluate_predictions,
    format_report,
)


class RejectionTests(unittest.TestCase):
    def test_predictions_below_threshold_become_unknown(self):
        result = apply_rejection(
            ['A', 'B', 'C'],
            confidences=[0.9, 0.4, 0.8],
            rejection_threshold=0.75,
        )

        self.assertEqual(result, ['A', UNKNOWN_LABEL, 'C'])

    def test_without_threshold_nothing_is_rejected(self):
        result = apply_rejection(['A', 'B'], confidences=[0.1, 0.2])
        self.assertEqual(result, ['A', 'B'])

    def test_mismatched_confidences_are_rejected_as_input_error(self):
        with self.assertRaises(ValueError):
            apply_rejection(['A', 'B'], confidences=[0.9], rejection_threshold=0.5)


class LatencyTests(unittest.TestCase):
    def test_latency_percentiles(self):
        stats = compute_latency_stats([10, 20, 30, 40])

        self.assertEqual(stats.count, 4)
        self.assertAlmostEqual(stats.mean_ms, 25.0)
        self.assertAlmostEqual(stats.p50_ms, 25.0)
        self.assertAlmostEqual(stats.max_ms, 40.0)

    def test_empty_latency_is_zeroed(self):
        stats = compute_latency_stats([])
        self.assertEqual(stats.count, 0)
        self.assertEqual(stats.mean_ms, 0.0)


class EvaluationReportTests(unittest.TestCase):
    def test_perfect_predictions(self):
        report = evaluate_predictions(['A', 'B', 'A'], ['A', 'B', 'A'])

        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(report.macro_f1, 1.0)
        self.assertEqual(report.rejection_rate, 0.0)
        self.assertEqual(report.labels, ['A', 'B'])

    def test_per_class_precision_recall_and_confusion_matrix(self):
        y_true = ['A', 'A', 'B', 'B']
        y_pred = ['A', 'B', 'B', 'B']

        report = evaluate_predictions(y_true, y_pred)

        self.assertEqual(report.accuracy, 0.75)

        metrics_a = report.per_class['A']
        self.assertEqual(metrics_a.support, 2)
        self.assertEqual(metrics_a.true_positives, 1)
        self.assertEqual(metrics_a.false_negatives, 1)
        self.assertEqual(metrics_a.precision, 1.0)
        self.assertEqual(metrics_a.recall, 0.5)

        metrics_b = report.per_class['B']
        self.assertEqual(metrics_b.recall, 1.0)
        self.assertAlmostEqual(metrics_b.precision, 2 / 3)

        expected = np.array([[1, 1], [0, 2]], dtype=np.int64)
        self.assertTrue(np.array_equal(report.confusion_matrix, expected))

    def test_rejection_is_counted_per_class_and_not_as_wrong_class(self):
        y_true = ['A', 'A', 'B', 'B']
        y_pred = ['A', 'C', 'B', 'B']
        confidences = [0.9, 0.2, 0.9, 0.1]

        report = evaluate_predictions(
            y_true, y_pred, confidences=confidences, rejection_threshold=0.75
        )

        self.assertEqual(report.rejection_threshold, 0.75)
        self.assertEqual(report.rejection_rate, 0.5)
        self.assertEqual(report.per_class['A'].rejected, 1)
        self.assertEqual(report.per_class['A'].rejection_rate, 0.5)
        self.assertEqual(report.per_class['B'].rejection_rate, 0.5)

        # A predição 'C' foi rejeitada, então C não vira falso positivo.
        self.assertNotIn('C', report.labels)
        self.assertIn(UNKNOWN_LABEL, report.labels)

        # Métricas macro ignoram a classe de rejeição.
        self.assertEqual(report.known_labels, ['A', 'B'])
        self.assertAlmostEqual(report.macro_recall, 0.5)

    def test_out_of_vocabulary_samples_are_scored_as_a_real_class(self):
        y_true = [UNKNOWN_LABEL, UNKNOWN_LABEL, 'A']
        y_pred = ['A', 'A', 'A']
        confidences = [0.1, 0.9, 0.9]

        report = evaluate_predictions(
            y_true, y_pred, confidences=confidences, rejection_threshold=0.75
        )

        unknown_metrics = report.per_class[UNKNOWN_LABEL]
        self.assertEqual(unknown_metrics.support, 2)
        # Apenas uma das duas amostras fora do vocabulário foi corretamente recusada.
        self.assertEqual(unknown_metrics.true_positives, 1)
        self.assertEqual(unknown_metrics.recall, 0.5)

    def test_explicit_label_order_is_respected_and_completed(self):
        report = evaluate_predictions(['B', 'A'], ['B', 'A'], labels=['B', 'A', 'C'])

        self.assertEqual(report.labels, ['B', 'A', 'C'])
        self.assertEqual(report.per_class['C'].support, 0)
        self.assertEqual(report.per_class['C'].f1, 0.0)

    def test_latency_is_attached_to_the_report(self):
        report = evaluate_predictions(['A', 'A'], ['A', 'A'], latencies_ms=[12.0, 18.0])

        self.assertIsNotNone(report.latency)
        self.assertAlmostEqual(report.latency.mean_ms, 15.0)

    def test_mismatched_lengths_and_empty_input_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_predictions(['A'], ['A', 'B'])
        with self.assertRaises(ValueError):
            evaluate_predictions([], [])

    def test_report_is_serializable_and_printable(self):
        report = evaluate_predictions(
            ['A', 'B'], ['A', 'B'], confidences=[0.9, 0.9], rejection_threshold=0.75,
            latencies_ms=[10.0, 12.0],
        )

        payload = report.to_dict()
        self.assertEqual(payload['confusion_matrix'], [[1, 0], [0, 1]])
        self.assertEqual(payload['labels'], ['A', 'B'])
        self.assertIsInstance(payload['latency'], dict)

        text = format_report(report)
        self.assertIn('Acurácia', text)
        self.assertIn('Matriz de confusão', text)


if __name__ == '__main__':
    unittest.main()
