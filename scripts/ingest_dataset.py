#!/usr/bin/env python3
"""Ingere uma base externa de Libras no dataset do LibrIA.

    python -m scripts.ingest_dataset data/archives/v-librasil \\
        --modality temporal --source-name v-librasil \\
        --label-map data/label_maps/v-librasil.json

Espera uma pasta por classe dentro do diretório de origem. O que não casar com
o vocabulário (``config/vocabulary.py``) é listado no fim do relatório para você
decidir se vale mapear ou ignorar.
"""

import argparse
import json
import os

from config.settings import STATIC_DATASET_DIR, TEMPORAL_DATASET_DIR
from config.data_sources import get_source
from config.vocabulary import MODALITY_STATIC, MODALITY_TEMPORAL
from src.dataset.video_ingest import (
    LABEL_FROM_DIR,
    LABEL_FROM_FILENAME,
    IngestOptions,
    ingest_directory,
    load_label_map,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Ingestão de bases externas no dataset LibrIA')
    parser.add_argument('source_dir', help='Diretório com os arquivos da base (uma pasta por sinal)')
    parser.add_argument('--modality', choices=[MODALITY_STATIC, MODALITY_TEMPORAL],
                        default=MODALITY_TEMPORAL, help='Fluxo de destino')
    parser.add_argument('--source-name', default=None,
                        help='Chave da fonte (ver `make sources`); vira source_dataset nos metadados')
    parser.add_argument('--dataset-dir', default=None, help='Sobrescreve o destino padrão')
    parser.add_argument('--label-map', default=None,
                        help='JSON {termo_da_base: LABEL_LIBRIA} para casar o vocabulário')
    parser.add_argument('--label-from', choices=[LABEL_FROM_DIR, LABEL_FROM_FILENAME],
                        default=LABEL_FROM_DIR,
                        help='De onde vem o sinal: pasta (padrão) ou nome do arquivo')
    parser.add_argument('--label-regex', default=None,
                        help='Regex sobre o caminho relativo com o grupo (?P<label>...) do sinal')
    parser.add_argument('--subject-pattern', default=None,
                        help='Regex sobre o caminho relativo com o grupo (?P<subject>...) do sinalizante')
    parser.add_argument('--default-subject', default=None,
                        help='subject_id quando o padrão não identifica a pessoa')
    parser.add_argument('--accept-unknown-labels', action='store_true',
                        help='Ingere também classes fora do vocabulário (cria diretórios novos)')
    parser.add_argument('--max-per-label', type=int, default=None,
                        help='Teto de amostras por classe nesta sessão')
    parser.add_argument('--frame-stride', type=int, default=1,
                        help='Processa 1 a cada N frames (acelera vídeos longos)')
    parser.add_argument('--static-frames-per-video', type=int, default=5,
                        help='Poses extraídas por vídeo no modo estático')
    parser.add_argument('--min-detection-ratio', type=float, default=0.4,
                        help='Fração mínima de frames com mão detectada para aceitar o vídeo')
    parser.add_argument('--no-mirror', action='store_true', help='Não gerar a amostra espelhada')
    parser.add_argument('--dry-run', action='store_true',
                        help='Só relata o que seria ingerido, sem escrever nada')
    parser.add_argument('--json', dest='json_path', default=None, help='Grava o relatório em JSON')
    parser.add_argument('--quiet', action='store_true', help='Não imprime o progresso arquivo a arquivo')
    return parser


def main() -> int:
    args = build_parser().parse_args()

    source_name = args.source_name or os.path.basename(os.path.normpath(args.source_dir))
    catalog_entry = get_source(source_name)

    options = IngestOptions(
        source_name=source_name,
        modality=args.modality,
        source_uri=catalog_entry.url if catalog_entry else '',
        license=catalog_entry.license if catalog_entry else '',
        default_subject=args.default_subject or source_name,
        min_detection_ratio=args.min_detection_ratio,
        static_frames_per_video=args.static_frames_per_video,
        max_samples_per_label=args.max_per_label,
        frame_stride=max(1, args.frame_stride),
        save_mirrored=not args.no_mirror,
        dry_run=args.dry_run,
    )

    dataset_dir = args.dataset_dir or (
        TEMPORAL_DATASET_DIR if args.modality == MODALITY_TEMPORAL else STATIC_DATASET_DIR
    )

    report = ingest_directory(
        args.source_dir,
        options,
        dataset_dir=dataset_dir,
        label_map=load_label_map(args.label_map),
        subject_pattern=args.subject_pattern,
        only_vocabulary=not args.accept_unknown_labels,
        label_from=args.label_from,
        label_pattern=args.label_regex,
        progress=None if args.quiet else print,
    )

    print()
    print(report.summary())
    if args.dry_run:
        print('  (dry-run: nada foi gravado)')

    if args.json_path:
        os.makedirs(os.path.dirname(args.json_path) or '.', exist_ok=True)
        with open(args.json_path, 'w', encoding='utf-8') as file_obj:
            json.dump(report.to_dict(), file_obj, indent=2, ensure_ascii=False)
        print(f'\nRelatório salvo em: {args.json_path}')

    return 0 if not report.failed or report.ingested else 1


if __name__ == '__main__':
    raise SystemExit(main())
