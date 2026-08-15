"""
Ingestão de bases externas
==========================

Converte vídeos e imagens de bases públicas de Libras (ver
``config/data_sources.py``) no mesmo formato que a coleta por webcam produz:
``sample_XXX.npy`` / ``seq_XXX.npy`` mais o ``.json`` de metadados. A partir daí
treino, avaliação e export embedded não distinguem a origem — só o campo
``source_dataset`` guarda a proveniência.

É esta via que torna a gravação manual excepcional: uma classe já coberta por
uma base externa não precisa de sessão de webcam nenhuma.

Duas garantias importantes:

* **Idempotência** — cada arquivo de origem já processado fica registrado em
  ``.ingest_state.json`` dentro do dataset. Rodar de novo não duplica amostras
  nem reprocessa vídeo (extrair landmarks é a parte cara).
* **Vocabulário fechado** — por padrão só entram classes que existem em
  ``config/vocabulary.py``. Bases com milhares de termos (V-LIBRASIL) não
  poluem o dataset com classes que nenhum modelo do projeto usa.
"""

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    COLLECTION_CONFIG,
    FEATURE_DIMENSION,
    FEATURE_MODE,
    LSTM_CONFIG,
    STATIC_DATASET_DIR,
    TEMPORAL_DATASET_DIR,
)
from config.vocabulary import (
    MODALITY_STATIC,
    MODALITY_TEMPORAL,
    UNKNOWN_LABEL,
    get_sign_type,
    is_known_label,
)
from src.dataset.landmark_storage import (
    next_sample_index,
    persist_sample,
    storage_shape,
    write_manifest,
)
from src.dataset.sample_metadata import UNSPECIFIED, SampleMetadata
from utils.helpers import extract_landmarks_by_mode, preprocess_frame

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except (ImportError, RuntimeError) as error:  # pragma: no cover - depende do host
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_ERROR = error

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.mpg', '.mpeg'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

STATE_FILENAME = '.ingest_state.json'


# ---------------------------------------------------------------------------
# Normalização de rótulos
# ---------------------------------------------------------------------------

def normalize_label(raw_label: str) -> str:
    """Converte o rótulo da base externa no formato de diretório do projeto.

    ``"Tudo bem?"`` vira ``TUDO_BEM``: maiúsculas, sem acento, sem espaço. As
    bases usam grafias variadas para o mesmo sinal, então normalizar aqui evita
    criar duas pastas para a mesma classe.
    """
    text = unicodedata.normalize('NFKD', str(raw_label).strip())
    text = ''.join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^\w\s-]", '', text, flags=re.ASCII)
    text = re.sub(r'[\s-]+', '_', text).strip('_')
    return text.upper()


def load_label_map(path: Optional[str]) -> Dict[str, str]:
    """Carrega o mapa ``termo da base -> label do LibrIA`` de um JSON.

    As chaves são normalizadas na leitura, então tanto ``"tudo bem"`` quanto
    ``"TUDO_BEM"`` casam com o mesmo termo de origem.
    """
    if not path:
        return {}

    with open(path, 'r', encoding='utf-8') as file_obj:
        payload = json.load(file_obj)

    if not isinstance(payload, dict):
        raise ValueError(f'Mapa de labels deve ser um objeto JSON: {path}')

    return {normalize_label(key): normalize_label(value) for key, value in payload.items()}


# ---------------------------------------------------------------------------
# Descoberta de arquivos
# ---------------------------------------------------------------------------

@dataclass
class SourceItem:
    """Um arquivo de origem já associado a uma classe do vocabulário."""

    path: str
    label: str
    raw_label: str
    subject_id: str


LABEL_FROM_DIR = 'dir'
LABEL_FROM_FILENAME = 'filename'


def _label_from_path(
    path: str,
    root: str,
    label_from: str = LABEL_FROM_DIR,
    label_pattern: Optional[re.Pattern] = None,
) -> Optional[str]:
    """Deriva o rótulo bruto do caminho.

    Três convenções cobrem as bases reais: uma pasta por sinal (o caso comum),
    o sinal no nome do arquivo (MINDS-Libras empacota por sinalizante, não por
    sinal) ou um regex explícito quando o nome mistura sinal, pessoa e take.
    """
    relative = os.path.relpath(path, root)

    if label_pattern is not None:
        match = label_pattern.search(relative)
        if not match:
            return None
        groups = match.groupdict()
        return groups.get('label') or (match.group(1) if match.groups() else None)

    parts = relative.split(os.sep)
    if label_from == LABEL_FROM_DIR and len(parts) > 1:
        return parts[0]
    return os.path.splitext(parts[-1])[0]


def _subject_from_path(path: str, root: str, pattern: Optional[re.Pattern], fallback: str) -> str:
    if pattern is None:
        return fallback
    match = pattern.search(os.path.relpath(path, root))
    if not match:
        return fallback
    groups = match.groupdict()
    return groups.get('subject') or (match.group(1) if match.groups() else fallback)


def discover_source_items(
    root: str,
    modality: str,
    label_map: Optional[Dict[str, str]] = None,
    subject_pattern: Optional[str] = None,
    default_subject: str = UNSPECIFIED,
    only_vocabulary: bool = True,
    label_from: str = LABEL_FROM_DIR,
    label_pattern: Optional[str] = None,
) -> Tuple[List[SourceItem], Dict[str, int]]:
    """Varre ``root`` e devolve os arquivos ingeríveis e os termos descartados."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f'Diretório de origem não encontrado: {root}')

    label_map = label_map or {}
    extensions = VIDEO_EXTENSIONS if modality == MODALITY_TEMPORAL else IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    compiled_subject = re.compile(subject_pattern) if subject_pattern else None
    compiled_label = re.compile(label_pattern) if label_pattern else None

    items: List[SourceItem] = []
    skipped: Dict[str, int] = {}

    for directory, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if os.path.splitext(filename)[1].lower() not in extensions:
                continue

            path = os.path.join(directory, filename)
            raw_label = _label_from_path(path, root, label_from, compiled_label)
            if raw_label is None:
                skipped['<sem rótulo no caminho>'] = skipped.get('<sem rótulo no caminho>', 0) + 1
                continue

            label = label_map.get(normalize_label(raw_label), normalize_label(raw_label))

            if only_vocabulary and label != UNKNOWN_LABEL and not is_known_label(label):
                skipped[raw_label] = skipped.get(raw_label, 0) + 1
                continue

            items.append(
                SourceItem(
                    path=path,
                    label=label,
                    raw_label=raw_label,
                    subject_id=_subject_from_path(path, root, compiled_subject, default_subject),
                )
            )

    return items, skipped


# ---------------------------------------------------------------------------
# Estado da ingestão (idempotência)
# ---------------------------------------------------------------------------

class IngestState:
    """Registro do que já foi ingerido, para não reprocessar nem duplicar."""

    def __init__(self, dataset_dir: str):
        self.path = os.path.join(dataset_dir, STATE_FILENAME)
        self.entries: Dict[str, Dict[str, object]] = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as file_obj:
                    payload = json.load(file_obj)
                if isinstance(payload, dict):
                    self.entries = payload.get('entries', {})
            except (OSError, json.JSONDecodeError):
                # Estado corrompido não pode travar a ingestão: reconstruímos.
                self.entries = {}

    @staticmethod
    def key_for(path: str) -> str:
        """Identidade do arquivo de origem: caminho + tamanho.

        Tamanho entra para que um arquivo trocado (download refeito, versão
        nova da base) seja reprocessado em vez de silenciosamente ignorado.
        """
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1
        return f'{os.path.abspath(path)}::{size}'

    def already_ingested(self, path: str) -> bool:
        return self.key_for(path) in self.entries

    def record(self, path: str, label: str, sample_name: str, source_dataset: str) -> None:
        self.entries[self.key_for(path)] = {
            'label': label,
            'sample': sample_name,
            'source_dataset': source_dataset,
        }

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as file_obj:
            json.dump({'entries': self.entries}, file_obj, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Extração de landmarks
# ---------------------------------------------------------------------------

def build_hands(static_image_mode: bool):
    if not MEDIAPIPE_AVAILABLE:
        raise RuntimeError(
            'MediaPipe não disponível para ingestão. '
            f'Motivo: {type(MEDIAPIPE_ERROR).__name__}: {MEDIAPIPE_ERROR}'
        )

    return mp.solutions.hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=1,
        min_detection_confidence=COLLECTION_CONFIG['min_detection_confidence'],
        min_tracking_confidence=COLLECTION_CONFIG['min_tracking_confidence'],
    )


def extract_frame_sample(frame: np.ndarray, hands, calibration=None) -> Optional[np.ndarray]:
    """Landmarks de um frame, no shape gravado em disco, ou ``None``."""
    frame = preprocess_frame(frame, calibration)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return None

    features = extract_landmarks_by_mode(results.multi_hand_landmarks[0].landmark, FEATURE_MODE)
    if features is None or np.asarray(features).size != FEATURE_DIMENSION:
        return None
    return storage_shape(features)


def resample_sequence(frames: List[np.ndarray], length: int) -> np.ndarray:
    """Reamostra uniformemente para o comprimento fixo esperado pela LSTM.

    Vídeos externos têm duração e fps arbitrários; o modelo espera sempre
    ``LSTM_CONFIG['sequence_length']`` passos. Amostrar índices igualmente
    espaçados preserva a trajetória do sinal sem depender do fps da origem.
    """
    if not frames:
        raise ValueError('Sequência vazia: nenhum frame com mão detectada')

    indices = np.linspace(0, len(frames) - 1, num=length)
    return np.asarray([frames[int(round(index))] for index in indices], dtype=np.float32)


def read_video_landmarks(
    path: str,
    hands,
    calibration=None,
    frame_stride: int = 1,
) -> Tuple[List[np.ndarray], int]:
    """Percorre o vídeo e devolve os frames com mão detectada e o total lido."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f'Não foi possível abrir o vídeo: {path}')

    landmarks: List[np.ndarray] = []
    total_frames = 0
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            if frame_stride > 1 and total_frames % frame_stride:
                total_frames += 1
                continue
            total_frames += 1

            sample = extract_frame_sample(frame, hands, calibration)
            if sample is not None:
                landmarks.append(sample)
    finally:
        capture.release()

    return landmarks, total_frames


# ---------------------------------------------------------------------------
# Ingestão
# ---------------------------------------------------------------------------

@dataclass
class IngestOptions:
    """Parâmetros de uma sessão de ingestão."""

    source_name: str
    modality: str = MODALITY_TEMPORAL
    source_uri: str = ''
    license: str = ''
    environment: str = 'base_externa'
    camera_id: str = UNSPECIFIED
    dominant_hand: str = UNSPECIFIED
    default_subject: str = UNSPECIFIED
    sequence_length: int = LSTM_CONFIG['sequence_length']
    # Vídeo curto ou com a mão fora de quadro não vira sinal utilizável.
    min_valid_frames: int = 8
    min_detection_ratio: float = 0.4
    # Quantos frames virar amostra estática por vídeo (imagens sempre viram 1).
    static_frames_per_video: int = 5
    max_samples_per_label: Optional[int] = None
    frame_stride: int = 1
    save_mirrored: bool = True
    dry_run: bool = False


@dataclass
class IngestReport:
    """O que a sessão produziu — impresso ao fim e gravado em JSON."""

    source_name: str = ''
    modality: str = ''
    dataset_dir: str = ''
    discovered: int = 0
    ingested: int = 0
    skipped_existing: int = 0
    skipped_quality: int = 0
    failed: int = 0
    per_label: Dict[str, int] = field(default_factory=dict)
    unmapped_labels: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            'source_name': self.source_name,
            'modality': self.modality,
            'dataset_dir': self.dataset_dir,
            'discovered': self.discovered,
            'ingested': self.ingested,
            'skipped_existing': self.skipped_existing,
            'skipped_quality': self.skipped_quality,
            'failed': self.failed,
            'per_label': self.per_label,
            'unmapped_labels': self.unmapped_labels,
            'errors': self.errors,
        }

    def summary(self) -> str:
        lines = [
            f'Ingestão de {self.source_name} ({self.modality}) → {self.dataset_dir}',
            f'  arquivos encontrados: {self.discovered}',
            f'  amostras criadas:     {self.ingested}',
            f'  já ingeridos antes:   {self.skipped_existing}',
            f'  descartados (qualidade): {self.skipped_quality}',
            f'  falhas:               {self.failed}',
        ]
        if self.per_label:
            detalhe = ', '.join(f'{label}={count}' for label, count in sorted(self.per_label.items()))
            lines.append(f'  por classe: {detalhe}')
        if self.unmapped_labels:
            top = sorted(self.unmapped_labels.items(), key=lambda item: -item[1])[:10]
            lines.append(
                '  termos fora do vocabulário (use LABEL_MAP para aproveitá-los): '
                + ', '.join(f'{label}({count})' for label, count in top)
            )
        for error in self.errors[:5]:
            lines.append(f'  erro: {error}')
        return '\n'.join(lines)


def _build_metadata(
    label: str,
    item: SourceItem,
    options: IngestOptions,
    duration_seconds: Optional[float],
    quality: Optional[float],
    sequence_length: Optional[int],
) -> SampleMetadata:
    return SampleMetadata(
        label=label,
        modality=options.modality,
        sign_type=get_sign_type(label) or UNSPECIFIED,
        subject_id=item.subject_id,
        camera_id=options.camera_id,
        environment=options.environment,
        dominant_hand=options.dominant_hand,
        capture_hand=UNSPECIFIED,
        duration_seconds=duration_seconds,
        quality=quality,
        feature_mode=FEATURE_MODE,
        feature_dimension=FEATURE_DIMENSION,
        sequence_length=sequence_length,
        source_sample=os.path.basename(item.path),
        source_dataset=options.source_name,
        source_uri=options.source_uri or None,
        license=options.license or None,
    )


def _ingest_temporal_item(
    item: SourceItem,
    label_dir: str,
    hands,
    calibration,
    options: IngestOptions,
    report: IngestReport,
) -> bool:
    landmarks, total_frames = read_video_landmarks(item.path, hands, calibration, options.frame_stride)

    detection_ratio = (len(landmarks) / total_frames) if total_frames else 0.0
    if len(landmarks) < options.min_valid_frames or detection_ratio < options.min_detection_ratio:
        report.skipped_quality += 1
        return False

    sequence = resample_sequence(landmarks, options.sequence_length)
    metadata = _build_metadata(
        item.label,
        item,
        options,
        duration_seconds=None,
        quality=float(detection_ratio),
        sequence_length=options.sequence_length,
    )

    index = next_sample_index(label_dir, 'seq_')
    persist_sample(
        label_dir,
        f'seq_{index:03d}',
        sequence,
        metadata,
        save_mirrored=options.save_mirrored,
    )
    return True


def _static_frames_from_item(item: SourceItem, hands, calibration, options: IngestOptions) -> List[np.ndarray]:
    extension = os.path.splitext(item.path)[1].lower()

    if extension in IMAGE_EXTENSIONS:
        frame = cv2.imread(item.path)
        if frame is None:
            raise RuntimeError(f'Imagem ilegível: {item.path}')
        sample = extract_frame_sample(frame, hands, calibration)
        return [sample] if sample is not None else []

    # Vídeo em modo estático: pega poses espalhadas ao longo do clipe, o que dá
    # variação de ângulo sem gravar nada a mais.
    landmarks, _ = read_video_landmarks(item.path, hands, calibration, options.frame_stride)
    if not landmarks:
        return []
    count = min(options.static_frames_per_video, len(landmarks))
    indices = np.linspace(0, len(landmarks) - 1, num=count)
    return [landmarks[int(round(index))] for index in indices]


def _ingest_static_item(
    item: SourceItem,
    label_dir: str,
    hands,
    calibration,
    options: IngestOptions,
    report: IngestReport,
) -> int:
    frames = _static_frames_from_item(item, hands, calibration, options)
    if not frames:
        report.skipped_quality += 1
        return 0

    created = 0
    for frame_sample in frames:
        metadata = _build_metadata(
            item.label, item, options, duration_seconds=None, quality=None, sequence_length=None
        )
        index = next_sample_index(label_dir, 'sample_')
        persist_sample(
            label_dir,
            f'sample_{index:03d}',
            np.asarray(frame_sample, dtype=np.float32),
            metadata,
            save_mirrored=options.save_mirrored,
        )
        created += 1
    return created


def ingest_directory(
    root: str,
    options: IngestOptions,
    dataset_dir: Optional[str] = None,
    label_map: Optional[Dict[str, str]] = None,
    subject_pattern: Optional[str] = None,
    only_vocabulary: bool = True,
    label_from: str = LABEL_FROM_DIR,
    label_pattern: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> IngestReport:
    """Ingere ``root`` inteiro no dataset, pulando o que já foi processado."""
    dataset_dir = dataset_dir or (
        TEMPORAL_DATASET_DIR if options.modality == MODALITY_TEMPORAL else STATIC_DATASET_DIR
    )
    emit = progress or (lambda message: None)

    items, skipped = discover_source_items(
        root,
        options.modality,
        label_map=label_map,
        subject_pattern=subject_pattern,
        default_subject=options.default_subject or options.source_name,
        only_vocabulary=only_vocabulary,
        label_from=label_from,
        label_pattern=label_pattern,
    )

    report = IngestReport(
        source_name=options.source_name,
        modality=options.modality,
        dataset_dir=dataset_dir,
        discovered=len(items),
        unmapped_labels=skipped,
    )

    if options.dry_run or not items:
        return report

    os.makedirs(dataset_dir, exist_ok=True)
    state = IngestState(dataset_dir)
    calibration = None  # bases externas vêm de câmeras desconhecidas
    hands = build_hands(static_image_mode=options.modality == MODALITY_STATIC)

    try:
        for position, item in enumerate(items, start=1):
            if state.already_ingested(item.path):
                report.skipped_existing += 1
                continue

            limit = options.max_samples_per_label
            if limit is not None and report.per_label.get(item.label, 0) >= limit:
                continue

            label_dir = os.path.join(dataset_dir, item.label)
            os.makedirs(label_dir, exist_ok=True)
            emit(f'[{position}/{len(items)}] {item.label} ← {os.path.basename(item.path)}')

            try:
                if options.modality == MODALITY_TEMPORAL:
                    created = 1 if _ingest_temporal_item(
                        item, label_dir, hands, calibration, options, report
                    ) else 0
                else:
                    created = _ingest_static_item(item, label_dir, hands, calibration, options, report)
            except Exception as error:  # arquivo ruim não pode abortar a base inteira
                report.failed += 1
                report.errors.append(f'{item.path}: {error}')
                continue

            if not created:
                continue

            report.ingested += created
            report.per_label[item.label] = report.per_label.get(item.label, 0) + created
            state.record(item.path, item.label, f'{item.label}/{created}', options.source_name)
            state.save()
    finally:
        hands.close()

    if report.per_label:
        write_manifest(
            options.modality,
            sorted(report.per_label),
            dataset_dir,
            options.max_samples_per_label or 0,
            options.sequence_length if options.modality == MODALITY_TEMPORAL else None,
        )

    return report


def iter_ingestable_labels(root: str, modality: str, label_map: Optional[Dict[str, str]] = None) -> Iterator[str]:
    """Labels do vocabulário que ``root`` consegue cobrir — útil antes de gravar."""
    items, _ = discover_source_items(root, modality, label_map=label_map)
    seen = set()
    for item in items:
        if item.label not in seen:
            seen.add(item.label)
            yield item.label
