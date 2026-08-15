#!/usr/bin/env python3
"""Relatório de cobertura do dataset: vocabulário, metadados e divisão por pessoa."""

import argparse
import json
import os
from typing import Dict, List

from config.settings import (
    COLLECTION_CONFIG,
    EVALUATION_CONFIG,
    STATIC_DATASET_DIR,
    TEMPORAL_DATASET_DIR,
)
from config.vocabulary import MODALITY_STATIC, MODALITY_TEMPORAL, get_labels
from src.dataset.coverage import dataset_gaps
from src.dataset.sample_metadata import collect_dataset_metadata, metadata_coverage
from src.evaluation.dataset_splits import SPLIT_NAMES, split_metadata_by_person

TARGET_PER_LABEL = {
    MODALITY_STATIC: COLLECTION_CONFIG['static_samples_per_label'],
    MODALITY_TEMPORAL: COLLECTION_CONFIG['temporal_samples_per_label'],
}


def _vocabulary_coverage(dataset_dir: str, modality: str) -> Dict[str, int]:
    """Quantas amostras existem para cada label esperada nesta modalidade."""
    counts = {}
    for label in get_labels(modality=modality):
        label_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_dir):
            counts[label] = 0
            continue
        counts[label] = len([name for name in os.listdir(label_dir) if name.endswith('.npy')])
    return counts


def _report_for(dataset_dir: str, modality: str, allow_empty_splits: bool) -> Dict[str, object]:
    coverage = metadata_coverage(dataset_dir)
    vocabulary = _vocabulary_coverage(dataset_dir, modality)
    metadata_list = collect_dataset_metadata(dataset_dir)

    split_summary = None
    split_error = None
    if metadata_list:
        try:
            split = split_metadata_by_person(
                metadata_list,
                ratios=EVALUATION_CONFIG['split_ratios'],
                allow_empty_splits=allow_empty_splits,
            )
            split_summary = {
                'summary': split.summary(),
                'subjects': {name: split.subjects.get(name, []) for name in SPLIT_NAMES},
            }
        except ValueError as error:
            split_error = str(error)

    return {
        'dataset_dir': dataset_dir,
        'modality': modality,
        'metadata_coverage': coverage,
        'vocabulary_counts': vocabulary,
        'missing_labels': sorted(label for label, count in vocabulary.items() if count == 0),
        # O que ainda falta e de onde veio o que já existe: é isso que decide se
        # a próxima sessão é de webcam ou de ingestão (`make sources`).
        'gaps': dataset_gaps(dataset_dir, get_labels(modality=modality), TARGET_PER_LABEL[modality]),
        'split': split_summary,
        'split_error': split_error,
    }


def _print_report(report: Dict[str, object]) -> None:
    coverage = report['metadata_coverage']
    print(f"\n=== {report['modality'].upper()} — {report['dataset_dir']} ===")
    print(f"Amostras: {coverage['total_samples']}")
    print(
        f"Com metadados: {coverage['annotated_samples']} "
        f"({coverage['coverage'] * 100:.1f}%)"
    )
    print(f"Pessoas: {', '.join(coverage['subjects']) or '-'}")
    print(f"Ambientes: {', '.join(coverage['environments']) or '-'}")
    print(f"Câmeras: {', '.join(coverage['cameras']) or '-'}")

    missing: List[str] = report['missing_labels']
    print(f"Labels do vocabulário sem amostras ({len(missing)}): {', '.join(missing) or '-'}")

    gaps = report['gaps']
    origens = sorted({
        origin
        for counts in gaps['sources_per_label'].values()
        for origin in counts
    })
    print(f"Origens das amostras: {', '.join(origens) or '-'}")

    pending = gaps['pending_labels']
    if pending:
        detalhe = ', '.join(f'{label}(-{count})' for label, count in sorted(pending.items()))
        print(f"Faltando para a meta de {gaps['target_per_label']}/classe ({len(pending)}): {detalhe}")
        print("  → antes de gravar, veja se alguma base pública cobre: make sources")
    else:
        print(f"Meta de {gaps['target_per_label']} amostras por classe atingida.")

    single = gaps['single_source_labels']
    if single:
        print(f"Classes com uma única origem ({len(single)}): risco de viés de pessoa/câmera.")

    if report['split_error']:
        print(f"Divisão por pessoa indisponível: {report['split_error']}")
    elif report['split']:
        for name in SPLIT_NAMES:
            stats = report['split']['summary'][name]
            members = ', '.join(report['split']['subjects'][name]) or '-'
            print(f"  {name:<11} {stats['samples']:>5} amostras | {stats['subjects']} pessoas: {members}")


def main():
    parser = argparse.ArgumentParser(description='Relatório de cobertura do dataset LibrIA')
    parser.add_argument('--static-dir', default=STATIC_DATASET_DIR)
    parser.add_argument('--temporal-dir', default=TEMPORAL_DATASET_DIR)
    parser.add_argument('--allow-empty-splits', action='store_true',
                        help='Permite conjuntos vazios quando há poucas pessoas coletadas')
    parser.add_argument('--json', dest='json_path', default=None,
                        help='Grava o relatório completo em JSON no caminho informado')
    args = parser.parse_args()

    reports = [
        _report_for(args.static_dir, MODALITY_STATIC, args.allow_empty_splits),
        _report_for(args.temporal_dir, MODALITY_TEMPORAL, args.allow_empty_splits),
    ]

    for report in reports:
        _print_report(report)

    if args.json_path:
        os.makedirs(os.path.dirname(args.json_path) or '.', exist_ok=True)
        with open(args.json_path, 'w', encoding='utf-8') as file_obj:
            json.dump(reports, file_obj, indent=2, ensure_ascii=False)
        print(f'\nRelatório JSON salvo em: {args.json_path}')


if __name__ == '__main__':
    main()
