"""
Cobertura do dataset e plano de coleta
======================================

Regra do projeto: só se grava na webcam o que nenhuma outra fonte cobre. Este
módulo responde "o que ainda falta?" olhando o disco, e é a base tanto do
relatório (``make report``) quanto da coleta dirigida por lacunas
(``make collect``), que pula as classes já completas.

Conta-se apenas a amostra original: os arquivos ``*_mirror.npy`` são
augmentação derivada e contá-los faria uma classe pela metade parecer pronta.
"""

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List

from src.dataset.sample_metadata import read_metadata

MIRROR_SUFFIX = '_mirror'


def _is_original_sample(filename: str) -> bool:
    if not filename.endswith('.npy'):
        return False
    return not os.path.splitext(filename)[0].endswith(MIRROR_SUFFIX)


def count_label_samples(dataset_dir: str, label: str) -> int:
    """Quantas amostras originais existem para uma classe."""
    label_dir = os.path.join(dataset_dir, label)
    if not os.path.isdir(label_dir):
        return 0
    return len([name for name in os.listdir(label_dir) if _is_original_sample(name)])


def count_label_sources(dataset_dir: str, label: str) -> Dict[str, int]:
    """Quantas amostras vieram de cada origem (webcam local ou base externa)."""
    label_dir = os.path.join(dataset_dir, label)
    counts: Dict[str, int] = {}
    if not os.path.isdir(label_dir):
        return counts

    for filename in sorted(os.listdir(label_dir)):
        if not _is_original_sample(filename):
            continue
        metadata = read_metadata(os.path.join(label_dir, filename))
        origin = (metadata.source_dataset if metadata else None) or 'coleta_local'
        counts[origin] = counts.get(origin, 0) + 1
    return counts


@dataclass
class LabelCoverage:
    """Situação de uma classe do vocabulário no disco."""

    label: str
    samples: int
    target: int
    sources: Dict[str, int]

    @property
    def missing(self) -> int:
        return max(self.target - self.samples, 0)

    @property
    def complete(self) -> bool:
        return self.missing == 0

    @property
    def only_local(self) -> bool:
        """Classe coberta, mas por uma única origem — ainda frágil a viés."""
        return len(self.sources) <= 1


def coverage_for(dataset_dir: str, labels: Iterable[str], target: int) -> List[LabelCoverage]:
    """Cobertura de cada label pedida, na ordem em que foi pedida."""
    return [
        LabelCoverage(
            label=label,
            samples=count_label_samples(dataset_dir, label),
            target=target,
            sources=count_label_sources(dataset_dir, label),
        )
        for label in labels
    ]


def pending_labels(dataset_dir: str, labels: Iterable[str], target: int) -> List[str]:
    """Labels que ainda não atingiram a meta — o que sobra para a webcam."""
    return [item.label for item in coverage_for(dataset_dir, labels, target) if not item.complete]


def format_collection_plan(
    dataset_dir: str,
    labels: Iterable[str],
    target: int,
    modality: str,
) -> str:
    """Texto do plano de coleta, mostrando o que será pulado e por quê."""
    items = coverage_for(dataset_dir, labels, target)
    done = [item for item in items if item.complete]
    todo = [item for item in items if not item.complete]

    lines = [f'Plano de coleta {modality} — meta de {target} amostras por classe']
    if done:
        lines.append(
            f'  já completas ({len(done)}), serão puladas: '
            + ', '.join(item.label for item in done)
        )
    if todo:
        lines.append(f'  a coletar ({len(todo)}):')
        for item in todo:
            origem = ', '.join(f'{name}={count}' for name, count in sorted(item.sources.items()))
            detalhe = f' | já em disco: {origem}' if origem else ''
            lines.append(f'    {item.label:<12} faltam {item.missing:>3}{detalhe}')
    else:
        lines.append('  nada a coletar: todas as classes atingiram a meta.')
        lines.append('  Para ampliar sem gravar, veja `make sources`.')

    single_source = [item for item in done if item.only_local]
    if single_source:
        lines.append(
            f'  atenção: {len(single_source)} classes vêm de uma única origem — '
            'considere ingerir uma base externa (`make sources`) antes de confiar na acurácia.'
        )
    return '\n'.join(lines)


def dataset_gaps(dataset_dir: str, labels: Iterable[str], target: int) -> Dict[str, object]:
    """Resumo das lacunas, no formato usado pelo relatório JSON."""
    items = coverage_for(dataset_dir, labels, target)
    return {
        'target_per_label': target,
        'complete_labels': [item.label for item in items if item.complete],
        'pending_labels': {item.label: item.missing for item in items if not item.complete},
        'sources_per_label': {item.label: item.sources for item in items if item.sources},
        'single_source_labels': [item.label for item in items if item.complete and item.only_local],
    }


def resolve_labels_to_collect(
    dataset_dir: str,
    labels: Iterable[str],
    target: int,
    only_missing: bool = True,
    modality: str = '',
    printer=print,
) -> List[str]:
    """Aplica o plano: imprime o que foi decidido e devolve o que coletar."""
    labels = list(labels)
    printer(format_collection_plan(dataset_dir, labels, target, modality or 'do dataset'))
    if not only_missing:
        return labels
    return pending_labels(dataset_dir, labels, target)
