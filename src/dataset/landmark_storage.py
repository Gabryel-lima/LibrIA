"""
Gravação de amostras de landmarks
=================================

O formato em disco (``sample_XXX.npy`` / ``seq_XXX.npy``, o irmão ``_mirror`` e
o ``.json`` de metadados) é o contrato entre quem produz dados e quem treina.
Ele nasceu na coleta por webcam, mas a ingestão de bases externas
(``src/dataset/video_ingest.py``) precisa gravar exatamente igual — por isso a
lógica vive aqui, e não dentro do script de coleta.
"""

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from config.settings import FEATURE_DIMENSION, FEATURE_MODE
from src.dataset.sample_metadata import SampleMetadata, write_metadata

MIRROR_SUFFIX = '_mirror'


def storage_shape(features: np.ndarray) -> np.ndarray:
    """Devolve o array no shape gravado em disco: ``(21, 3)`` em wrist_relative."""
    array = np.asarray(features, dtype=np.float32)
    if FEATURE_MODE == 'wrist_relative' and array.size == FEATURE_DIMENSION and FEATURE_DIMENSION % 3 == 0:
        return array.reshape(-1, 3)
    return array


def mirror_landmark_sample(sample: np.ndarray) -> np.ndarray:
    """Espelha no eixo X: dobra o dataset cobrindo canhotos e destros."""
    mirrored = np.asarray(sample, dtype=np.float32).copy()

    if mirrored.ndim == 2 and mirrored.shape[1] == 3:
        if FEATURE_MODE == 'wrist_relative':
            mirrored[:, 0] = -mirrored[:, 0]
        else:
            mirrored[:, 0] = 1.0 - mirrored[:, 0]
        return mirrored

    flat = mirrored.reshape(-1)
    if FEATURE_MODE == 'wrist_relative':
        flat[0::3] = -flat[0::3]
    else:
        flat[0::3] = 1.0 - flat[0::3]
    return flat.reshape(mirrored.shape)


def persist_sample(
    label_dir: str,
    base_name: str,
    sample_array: np.ndarray,
    metadata: Optional[SampleMetadata],
    save_mirrored: bool = True,
    skip_existing_mirror: bool = False,
) -> int:
    """Grava a amostra, sua versão espelhada e os metadados de ambas.

    Retorna quantos arquivos ``.npy`` foram criados.
    """
    written = 0

    sample_path = os.path.join(label_dir, f'{base_name}.npy')
    np.save(sample_path, sample_array)
    written += 1
    if metadata is not None:
        write_metadata(sample_path, metadata)

    if not save_mirrored:
        return written

    mirrored_path = os.path.join(label_dir, f'{base_name}{MIRROR_SUFFIX}.npy')
    if skip_existing_mirror and os.path.exists(mirrored_path):
        return written

    np.save(mirrored_path, mirror_landmark_sample(sample_array))
    written += 1
    if metadata is not None:
        write_metadata(mirrored_path, metadata.mirrored_copy(f'{base_name}.npy'))

    return written


def next_sample_index(label_dir: str, prefix: str) -> int:
    """Próximo índice livre — permite retomar a coleta sem sobrescrever nada."""
    existing_indices = []
    for filename in os.listdir(label_dir):
        if not filename.startswith(prefix) or not filename.endswith('.npy'):
            continue
        stem = os.path.splitext(filename)[0]
        suffix = stem.split('_')[-1]
        if suffix.isdigit():
            existing_indices.append(int(suffix))
    return max(existing_indices, default=-1) + 1


def write_manifest(
    mode: str,
    labels: List[str],
    output_dir: str,
    sample_target: int,
    sequence_length: Optional[int],
    camera_calibrated: bool = False,
) -> str:
    """Atualiza o ``manifest.json`` do subconjunto e devolve seu caminho."""
    manifest_path = os.path.join(output_dir, 'manifest.json')
    samples: Dict[str, int] = {}
    for label in labels:
        label_dir = os.path.join(output_dir, label)
        if not os.path.isdir(label_dir):
            samples[label] = 0
            continue
        samples[label] = len([name for name in os.listdir(label_dir) if name.endswith('.npy')])

    payload = {
        'mode': mode,
        'feature_mode': FEATURE_MODE,
        'feature_dimension': FEATURE_DIMENSION,
        'sample_target': sample_target,
        'sequence_length': sequence_length,
        'camera_calibrated': camera_calibrated,
        'labels': labels,
        'counts': samples,
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(manifest_path, 'w', encoding='utf-8') as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=True)
    return manifest_path
