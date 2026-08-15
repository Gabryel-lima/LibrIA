#!/usr/bin/env python3
"""Catálogo e download das bases externas de Libras.

    python -m scripts.fetch_sources --list
    python -m scripts.fetch_sources minds-libras --filter '\\.mp4$' --limit 20

Só baixa automaticamente o que tem URL pública estável (hoje, o registro do
Zenodo do MINDS-Libras). Para as bases que exigem conta ou aceite de termos, o
comando imprime as instruções em vez de tentar contornar o acesso.
"""

import argparse
import os
import re
from typing import List, Optional

from config.data_sources import DataSource, get_source, list_sources
from config.settings import DATA_DIR

ARCHIVES_DIR = os.path.join(DATA_DIR, 'archives')
ZENODO_RECORD_RE = re.compile(r'zenodo\.org/(?:record|records)/(\d+)')


def _print_source(source: DataSource, verbose: bool = False) -> None:
    acesso = {
        'direct': 'download automático',
        'account': 'requer conta na plataforma',
        'request': 'requer solicitação aos autores',
    }[source.access]

    print(f'\n{source.key}')
    print(f'  {source.name} — {source.language.upper()} | {source.modality} | {source.content}')
    print(f'  acesso: {acesso} | licença: {source.license}')
    if source.classes:
        print(f'  classes: {source.classes} | sinalizantes: {source.signers or "?"} | tamanho: {source.size}')
    print(f'  url: {source.url}')
    if verbose and source.notes:
        print(f'  {source.notes}')
    if verbose and source.instructions:
        print('  como usar:')
        for step in source.instructions:
            print(f'    - {step}')


def list_command(args) -> int:
    sources = list_sources(
        language=args.language,
        modality=args.modality,
        automatable_only=args.automatable_only,
    )
    if not sources:
        print('Nenhuma fonte corresponde ao filtro.')
        return 0

    print(f'Fontes externas de dados ({len(sources)}):')
    for source in sources:
        _print_source(source, verbose=not args.short)

    print('\nDepois de baixar, ingira sem gravar nada na webcam:')
    print('  make ingest SOURCE_DIR=data/archives/<base> MODALITY=temporal SOURCE_NAME=<base>')
    return 0


def _zenodo_files(record_id: str) -> List[dict]:
    import requests

    response = requests.get(f'https://zenodo.org/api/records/{record_id}', timeout=60)
    response.raise_for_status()
    return response.json().get('files', [])


def _download(url: str, destination: str) -> None:
    import requests

    if os.path.exists(destination):
        print(f'  já existe, pulando: {os.path.basename(destination)}')
        return

    partial = destination + '.part'
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        total = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        with open(partial, 'wb') as file_obj:
            for chunk in response.iter_content(chunk_size=1 << 20):
                file_obj.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f'\r  {os.path.basename(destination)}: '
                          f'{downloaded / total:.0%}', end='', flush=True)
    print()
    os.replace(partial, destination)


def fetch_command(args) -> int:
    source = get_source(args.source)
    if source is None:
        print(f'Fonte desconhecida: {args.source}. Veja `--list`.')
        return 1

    if not source.automatable:
        print(f'{source.name} não permite download automático ({source.access}).')
        _print_source(source, verbose=True)
        return 1

    match = ZENODO_RECORD_RE.search(source.url)
    if not match:
        print(f'Sem estratégia de download implementada para {source.url}')
        return 1

    target_dir = args.output or os.path.join(ARCHIVES_DIR, source.key)
    os.makedirs(target_dir, exist_ok=True)

    files = _zenodo_files(match.group(1))
    pattern = re.compile(args.filter) if args.filter else None
    selected = [
        item for item in files
        if pattern is None or pattern.search(item.get('key', ''))
    ]
    if args.limit:
        selected = selected[: args.limit]

    total_bytes = sum(item.get('size', 0) for item in selected)
    print(f'{source.name}: {len(selected)} de {len(files)} arquivos '
          f'({total_bytes / 1e9:.1f} GB) → {target_dir}')
    print(f'Licença: {source.license}')

    if args.dry_run:
        for item in selected:
            print(f'  {item.get("key")} ({item.get("size", 0) / 1e6:.0f} MB)')
        print('(dry-run: nada foi baixado)')
        return 0

    for item in selected:
        url = item.get('links', {}).get('self')
        if not url:
            continue
        _download(url, os.path.join(target_dir, item['key']))

    print('\nPronto. Agora ingira sem gravar nada:')
    print(f'  make ingest SOURCE_DIR={target_dir} MODALITY={source.modality} SOURCE_NAME={source.key}')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Bases externas de Libras: catálogo e download')
    parser.add_argument('source', nargs='?', help='Chave da fonte a baixar')
    parser.add_argument('--list', action='store_true', help='Lista o catálogo')
    parser.add_argument('--short', action='store_true', help='Listagem compacta')
    parser.add_argument('--language', default=None, help='Filtra por língua (libras, asl)')
    parser.add_argument('--modality', default=None, help='Filtra por modalidade (static, temporal)')
    parser.add_argument('--automatable-only', action='store_true',
                        help='Só as fontes com download automático')
    parser.add_argument('--output', default=None, help='Diretório de destino do download')
    parser.add_argument('--filter', default=None, help='Regex sobre o nome dos arquivos a baixar')
    parser.add_argument('--limit', type=int, default=None, help='Baixa no máximo N arquivos')
    parser.add_argument('--dry-run', action='store_true', help='Mostra o que seria baixado')
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.source and not args.list:
        return fetch_command(args)
    return list_command(args)


if __name__ == '__main__':
    raise SystemExit(main())
