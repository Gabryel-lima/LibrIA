"""
Metadados por amostra do dataset
================================

Cada amostra (``sample_XXX.npy`` ou ``seq_XXX.npy``) ganha um arquivo JSON
irmão com o mesmo nome (``sample_XXX.json``). Guardar os metadados fora do
``.npy`` mantém compatibilidade total com os datasets já coletados: quem só lê
landmarks continua ignorando os ``.json``.

Os campos seguem a Fase 1 do plano arquitetural: pessoa, câmera, ambiente, mão
dominante, duração, classe e qualidade da captura. São eles que permitem
dividir treino/validação/teste por pessoa e medir métricas por ambiente.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

METADATA_SCHEMA_VERSION = 1
METADATA_EXTENSION = '.json'

# Valor usado quando a informação não foi declarada na coleta. Preferimos uma
# string explícita a ``None`` para que agrupamentos por pessoa/ambiente não
# colapsem silenciosamente amostras de origens diferentes.
UNSPECIFIED = 'desconhecido'

HAND_LEFT = 'left'
HAND_RIGHT = 'right'
HANDS = (HAND_LEFT, HAND_RIGHT)


def _normalize_hand(hand: Optional[str]) -> str:
    if hand is None:
        return UNSPECIFIED
    normalized = str(hand).strip().lower()
    if normalized in {'l', 'left', 'esquerda', 'e'}:
        return HAND_LEFT
    if normalized in {'r', 'right', 'direita', 'd'}:
        return HAND_RIGHT
    return UNSPECIFIED


def mirror_hand(hand: Optional[str]) -> str:
    """Inverte a lateralidade — usado ao registrar a amostra espelhada."""
    normalized = _normalize_hand(hand)
    if normalized == HAND_LEFT:
        return HAND_RIGHT
    if normalized == HAND_RIGHT:
        return HAND_LEFT
    return UNSPECIFIED


@dataclass
class SampleMetadata:
    """Metadados de uma única amostra coletada."""

    label: str
    modality: str
    sign_type: str = UNSPECIFIED
    subject_id: str = UNSPECIFIED
    camera_id: str = UNSPECIFIED
    environment: str = UNSPECIFIED
    dominant_hand: str = UNSPECIFIED
    capture_hand: str = UNSPECIFIED
    duration_seconds: Optional[float] = None
    quality: Optional[float] = None
    feature_mode: str = UNSPECIFIED
    feature_dimension: Optional[int] = None
    sequence_length: Optional[int] = None
    mirrored: bool = False
    source_sample: Optional[str] = None
    created_at: str = field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%S'))
    schema_version: int = METADATA_SCHEMA_VERSION

    def __post_init__(self):
        self.label = str(self.label).strip().upper()
        self.dominant_hand = _normalize_hand(self.dominant_hand)
        self.capture_hand = _normalize_hand(self.capture_hand)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'SampleMetadata':
        known_fields = {f for f in cls.__dataclass_fields__}
        filtered = {key: value for key, value in payload.items() if key in known_fields}
        return cls(**filtered)

    def mirrored_copy(self, source_sample: str) -> 'SampleMetadata':
        """Metadados da versão espelhada desta amostra."""
        mirrored = SampleMetadata.from_dict(self.to_dict())
        mirrored.mirrored = True
        mirrored.capture_hand = mirror_hand(self.capture_hand)
        mirrored.dominant_hand = mirror_hand(self.dominant_hand)
        mirrored.source_sample = source_sample
        return mirrored


def metadata_path_for(sample_path: str) -> str:
    """Caminho do JSON irmão de uma amostra ``.npy``."""
    stem, _ = os.path.splitext(sample_path)
    return stem + METADATA_EXTENSION


def write_metadata(sample_path: str, metadata: SampleMetadata) -> str:
    """Escreve o JSON irmão da amostra e retorna o caminho gravado."""
    path = metadata_path_for(sample_path)
    with open(path, 'w', encoding='utf-8') as file_obj:
        json.dump(metadata.to_dict(), file_obj, indent=2, ensure_ascii=False)
    return path


def read_metadata(sample_path: str) -> Optional[SampleMetadata]:
    """Lê os metadados de uma amostra, ou ``None`` se ela não tiver JSON irmão."""
    path = metadata_path_for(sample_path)
    if not os.path.exists(path):
        return None

    try:
        with open(path, 'r', encoding='utf-8') as file_obj:
            payload = json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    return SampleMetadata.from_dict(payload)


def iter_dataset_samples(dataset_dir: str) -> Iterator[Tuple[str, Optional[SampleMetadata]]]:
    """Percorre ``dataset_dir/<label>/*.npy`` devolvendo (caminho, metadados)."""
    if not os.path.isdir(dataset_dir):
        return

    for label in sorted(os.listdir(dataset_dir)):
        label_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_dir):
            continue

        for filename in sorted(os.listdir(label_dir)):
            if not filename.endswith('.npy'):
                continue
            sample_path = os.path.join(label_dir, filename)
            yield sample_path, read_metadata(sample_path)


def collect_dataset_metadata(dataset_dir: str) -> List[SampleMetadata]:
    """Retorna os metadados de todas as amostras que os possuem."""
    return [
        metadata
        for _, metadata in iter_dataset_samples(dataset_dir)
        if metadata is not None
    ]


def metadata_coverage(dataset_dir: str) -> Dict[str, Any]:
    """Resume quantas amostras já têm metadados e quais pessoas/ambientes existem."""
    total = 0
    annotated = 0
    subjects = set()
    environments = set()
    cameras = set()
    per_label: Dict[str, Dict[str, int]] = {}

    for sample_path, metadata in iter_dataset_samples(dataset_dir):
        total += 1
        label = os.path.basename(os.path.dirname(sample_path))
        counters = per_label.setdefault(label, {'total': 0, 'annotated': 0})
        counters['total'] += 1

        if metadata is None:
            continue

        annotated += 1
        counters['annotated'] += 1
        subjects.add(metadata.subject_id)
        environments.add(metadata.environment)
        cameras.add(metadata.camera_id)

    return {
        'dataset_dir': dataset_dir,
        'total_samples': total,
        'annotated_samples': annotated,
        'coverage': (annotated / total) if total else 0.0,
        'subjects': sorted(subjects),
        'environments': sorted(environments),
        'cameras': sorted(cameras),
        'per_label': per_label,
    }
