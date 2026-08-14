"""
Métricas de reconhecimento por classe
=====================================

Acurácia global esconde exatamente o que importa na Fase 1: classes raras,
sinais visualmente parecidos e a fração de predições que o sistema deveria ter
recusado. Este módulo produz precisão, recall, F1 e matriz de confusão por
classe, além de duas medidas específicas do produto:

* **taxa de rejeição por classe** - quanto de cada classe cai abaixo do limiar
  de confiança e vira :data:`config.vocabulary.UNKNOWN_LABEL`;
* **latência** - média e percentis, para validar uso em tempo real.

Uma predição rejeitada não conta como acerto nem como erro de outra classe:
ela é contabilizada na coluna da classe de rejeição, que é preferível a uma
tradução errada apresentada como certeza.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from config.vocabulary import UNKNOWN_LABEL


@dataclass
class ClassMetrics:
    """Métricas de uma única classe."""

    label: str
    support: int
    true_positives: int
    false_positives: int
    false_negatives: int
    rejected: int
    precision: float
    recall: float
    f1: float
    rejection_rate: float


@dataclass
class LatencyStats:
    """Resumo de latência em milissegundos."""

    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


@dataclass
class EvaluationReport:
    """Relatório completo de uma avaliação."""

    labels: List[str]
    confusion_matrix: np.ndarray
    per_class: Dict[str, ClassMetrics]
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    rejection_rate: float
    unknown_label: str = UNKNOWN_LABEL
    rejection_threshold: Optional[float] = None
    latency: Optional[LatencyStats] = None
    known_labels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            'labels': list(self.labels),
            'confusion_matrix': self.confusion_matrix.tolist(),
            'per_class': {
                label: vars(metrics).copy() for label, metrics in self.per_class.items()
            },
            'accuracy': self.accuracy,
            'macro_precision': self.macro_precision,
            'macro_recall': self.macro_recall,
            'macro_f1': self.macro_f1,
            'rejection_rate': self.rejection_rate,
            'unknown_label': self.unknown_label,
            'rejection_threshold': self.rejection_threshold,
            'latency': vars(self.latency).copy() if self.latency else None,
            'known_labels': list(self.known_labels),
        }


def apply_rejection(
    predictions: Sequence[str],
    confidences: Optional[Sequence[float]] = None,
    rejection_threshold: Optional[float] = None,
    unknown_label: str = UNKNOWN_LABEL,
) -> List[str]:
    """Substitui por ``unknown_label`` as predições abaixo do limiar de confiança."""
    resolved = [str(prediction) for prediction in predictions]

    if rejection_threshold is None or confidences is None:
        return resolved

    if len(confidences) != len(resolved):
        raise ValueError('confidences e predictions devem ter o mesmo tamanho')

    return [
        unknown_label if float(confidence) < rejection_threshold else prediction
        for prediction, confidence in zip(resolved, confidences)
    ]


def compute_latency_stats(latencies_ms: Sequence[float]) -> LatencyStats:
    """Resume latências em ms (média, p50, p95 e máximo)."""
    values = np.asarray(list(latencies_ms), dtype=np.float64)
    if values.size == 0:
        return LatencyStats(count=0, mean_ms=0.0, p50_ms=0.0, p95_ms=0.0, max_ms=0.0)

    return LatencyStats(
        count=int(values.size),
        mean_ms=float(values.mean()),
        p50_ms=float(np.percentile(values, 50)),
        p95_ms=float(np.percentile(values, 95)),
        max_ms=float(values.max()),
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: Optional[Sequence[float]] = None,
    rejection_threshold: Optional[float] = None,
    labels: Optional[Sequence[str]] = None,
    unknown_label: str = UNKNOWN_LABEL,
    latencies_ms: Optional[Sequence[float]] = None,
) -> EvaluationReport:
    """Avalia predições com rejeição explícita e métricas por classe.

    ``confidences`` e ``rejection_threshold`` são opcionais: sem eles, nenhuma
    predição é rejeitada e ``y_pred`` é usado como veio. As métricas macro
    ignoram a classe de rejeição, que é reportada separadamente por
    ``rejection_rate``.
    """
    true_labels = [str(label) for label in y_true]
    if len(true_labels) != len(y_pred):
        raise ValueError('y_true e y_pred devem ter o mesmo tamanho')
    if not true_labels:
        raise ValueError('Nenhuma amostra fornecida para avaliação')

    effective_pred = apply_rejection(y_pred, confidences, rejection_threshold, unknown_label)

    if labels is None:
        label_list = sorted(set(true_labels).union(effective_pred))
    else:
        label_list = list(dict.fromkeys(str(label) for label in labels))
        for label in sorted(set(true_labels).union(effective_pred)):
            if label not in label_list:
                label_list.append(label)

    label_index = {label: index for index, label in enumerate(label_list)}
    matrix = np.zeros((len(label_list), len(label_list)), dtype=np.int64)
    for true_label, predicted_label in zip(true_labels, effective_pred):
        matrix[label_index[true_label], label_index[predicted_label]] += 1

    unknown_index = label_index.get(unknown_label)
    per_class: Dict[str, ClassMetrics] = {}

    for label in label_list:
        index = label_index[label]
        support = int(matrix[index, :].sum())
        true_positives = int(matrix[index, index])
        false_positives = int(matrix[:, index].sum()) - true_positives
        false_negatives = support - true_positives
        rejected = (
            int(matrix[index, unknown_index])
            if unknown_index is not None and unknown_index != index
            else 0
        )

        precision = _safe_divide(true_positives, true_positives + false_positives)
        recall = _safe_divide(true_positives, support)
        f1 = _safe_divide(2 * precision * recall, precision + recall)

        per_class[label] = ClassMetrics(
            label=label,
            support=support,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            rejected=rejected,
            precision=precision,
            recall=recall,
            f1=f1,
            rejection_rate=_safe_divide(rejected, support),
        )

    known_labels = [label for label in label_list if label != unknown_label]
    macro_source = known_labels or label_list

    total = len(true_labels)
    correct = int(np.trace(matrix))
    total_rejected = sum(metrics.rejected for metrics in per_class.values())

    return EvaluationReport(
        labels=label_list,
        confusion_matrix=matrix,
        per_class=per_class,
        accuracy=_safe_divide(correct, total),
        macro_precision=float(np.mean([per_class[label].precision for label in macro_source])),
        macro_recall=float(np.mean([per_class[label].recall for label in macro_source])),
        macro_f1=float(np.mean([per_class[label].f1 for label in macro_source])),
        rejection_rate=_safe_divide(total_rejected, total),
        unknown_label=unknown_label,
        rejection_threshold=rejection_threshold,
        latency=compute_latency_stats(latencies_ms) if latencies_ms is not None else None,
        known_labels=known_labels,
    )


def format_report(report: EvaluationReport, max_confusion_labels: int = 30) -> str:
    """Formata o relatório para leitura no terminal."""
    lines = [
        f'Acurácia: {report.accuracy:.4f}',
        f'Precisão (macro): {report.macro_precision:.4f}',
        f'Recall (macro): {report.macro_recall:.4f}',
        f'F1 (macro): {report.macro_f1:.4f}',
        f'Taxa de rejeição: {report.rejection_rate:.4f}',
    ]

    if report.rejection_threshold is not None:
        lines.append(f'Limiar de rejeição: {report.rejection_threshold:.2f}')

    if report.latency is not None:
        latency = report.latency
        lines.append(
            f'Latência (ms): média {latency.mean_ms:.1f} | p50 {latency.p50_ms:.1f} | '
            f'p95 {latency.p95_ms:.1f} | máx {latency.max_ms:.1f}'
        )

    lines.append('')
    lines.append(f'{"classe":<16}{"sup":>6}{"prec":>8}{"rec":>8}{"f1":>8}{"rejeicao":>10}')
    for label in report.labels:
        metrics = report.per_class[label]
        lines.append(
            f'{label:<16}{metrics.support:>6}{metrics.precision:>8.3f}'
            f'{metrics.recall:>8.3f}{metrics.f1:>8.3f}{metrics.rejection_rate:>10.3f}'
        )

    if len(report.labels) <= max_confusion_labels:
        lines.append('')
        lines.append('Matriz de confusão (linha = verdadeiro, coluna = predito):')
        header = ' ' * 16 + ''.join(f'{label:>8}' for label in report.labels)
        lines.append(header)
        for row_index, label in enumerate(report.labels):
            row = ''.join(f'{value:>8}' for value in report.confusion_matrix[row_index])
            lines.append(f'{label:<16}{row}')

    return '\n'.join(lines)
