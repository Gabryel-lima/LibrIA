#!/usr/bin/env python3
"""Coleta unificada de dataset estático e temporal para o LibrIA."""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from config.settings import (
    CAMERA_CONFIG,
    COLLECTION_CONFIG,
    FEATURE_DIMENSION,
    FEATURE_MODE,
    FUNCTIONAL_LABELS,
    LEXICAL_LABELS,
    LSTM_CONFIG,
    STATIC_DATASET_DIR,
    STATIC_LABELS,
    TEMPORAL_DATASET_DIR,
    TEMPORAL_LABELS,
    TEMPORAL_VOCABULARY_LABELS,
)
from config.vocabulary import MODALITY_STATIC, MODALITY_TEMPORAL, UNKNOWN_LABEL, get_sign_type
from src.dataset.coverage import resolve_labels_to_collect
from src.dataset.landmark_storage import (
    next_sample_index as _next_sample_index,
    persist_sample as _persist_sample,
    storage_shape as _storage_shape,
    write_manifest,
)
from src.dataset.sample_metadata import UNSPECIFIED, SampleMetadata
from utils.helpers import extract_landmarks_by_mode, load_camera_calibration, preprocess_frame

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except (ImportError, RuntimeError) as error:
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_ERROR = error


WINDOW_NAME = 'LibrIA - Coleta Unificada'
CAPTURE_KEYS = {32, 13, 10}
QUIT_KEYS = {27, ord('q')}

VOCABULARY_CHOICES = ('alphabet', 'lexical', 'functional', 'all', 'unknown')


@dataclass
class CaptureContext:
    """Contexto da sessão de coleta, gravado como metadado de cada amostra."""

    subject_id: str = COLLECTION_CONFIG['default_subject_id']
    camera_id: str = COLLECTION_CONFIG['default_camera_id']
    environment: str = COLLECTION_CONFIG['default_environment']
    dominant_hand: str = COLLECTION_CONFIG['default_dominant_hand']
    write_metadata: bool = COLLECTION_CONFIG['write_sample_metadata']


DEFAULT_CAPTURE_CONTEXT = CaptureContext()


def _resolve_vocabulary_labels(selection: str, modality: str) -> List[str]:
    """Traduz a opção --vocabulary na lista de labels a coletar."""
    if selection == 'unknown':
        return [UNKNOWN_LABEL]

    if modality == MODALITY_STATIC:
        # Palavras e gestos funcionais são temporais; no modo estático só o
        # alfabeto (e a classe de rejeição) faz sentido.
        return list(STATIC_LABELS)

    if selection == 'alphabet':
        return list(TEMPORAL_LABELS)
    if selection == 'lexical':
        return list(LEXICAL_LABELS)
    if selection == 'functional':
        return list(FUNCTIONAL_LABELS)
    return list(TEMPORAL_VOCABULARY_LABELS)


def _draw_status_overlay(preview: np.ndarray, lines: Iterable[str], accent_color=(0, 255, 0)):
    for index, line in enumerate(lines):
        color = accent_color if index == 0 else (255, 255, 255)
        cv2.putText(
            preview,
            line,
            (10, 30 + (index * 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )


def _load_optional_calibration() -> Optional[Dict[str, np.ndarray]]:
    if not CAMERA_CONFIG['enabled']:
        return None
    return load_camera_calibration(
        CAMERA_CONFIG['camera_matrix_path'],
        CAMERA_CONFIG['dist_coeffs_path'],
    )


def _build_metadata(
    label: str,
    modality: str,
    context: CaptureContext,
    capture_hand: Optional[str] = None,
    quality: Optional[float] = None,
    duration_seconds: Optional[float] = None,
    sequence_length: Optional[int] = None,
) -> SampleMetadata:
    return SampleMetadata(
        label=label,
        modality=modality,
        sign_type=get_sign_type(label) or UNSPECIFIED,
        subject_id=context.subject_id,
        camera_id=context.camera_id,
        environment=context.environment,
        dominant_hand=context.dominant_hand,
        capture_hand=capture_hand or UNSPECIFIED,
        duration_seconds=duration_seconds,
        quality=quality,
        feature_mode=FEATURE_MODE,
        feature_dimension=FEATURE_DIMENSION,
        sequence_length=sequence_length,
    )


def _handedness_score(results) -> Optional[float]:
    if not getattr(results, 'multi_handedness', None):
        return None
    return float(results.multi_handedness[0].classification[0].score)


def _handedness_label(results) -> Optional[str]:
    if not getattr(results, 'multi_handedness', None):
        return None
    return str(results.multi_handedness[0].classification[0].label)


def _write_manifest(mode: str, labels: List[str], output_dir: str, sample_target: int, sequence_length: Optional[int]):
    write_manifest(
        mode,
        labels,
        output_dir,
        sample_target,
        sequence_length,
        camera_calibrated=_load_optional_calibration() is not None,
    )


def _extract_valid_sample(results) -> Optional[np.ndarray]:
    if not results.multi_hand_landmarks:
        return None

    if results.multi_handedness:
        score = results.multi_handedness[0].classification[0].score
        if score < COLLECTION_CONFIG['min_detection_confidence']:
            return None

    features = extract_landmarks_by_mode(results.multi_hand_landmarks[0].landmark, FEATURE_MODE)
    if features is None or np.asarray(features).size != FEATURE_DIMENSION:
        return None

    return _storage_shape(features)


def _extract_valid_sample_from_frame(frame: np.ndarray, hands) -> Optional[np.ndarray]:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    return _extract_valid_sample(results)


def _backfill_static_samples_from_frames(
    label_dir: str,
    calibration: Optional[Dict[str, np.ndarray]],
    extractor,
    save_mirrored: bool = False,
    context: Optional[CaptureContext] = None,
) -> int:
    generated_samples = 0
    label = os.path.basename(os.path.normpath(label_dir))

    for filename in sorted(os.listdir(label_dir)):
        name, extension = os.path.splitext(filename)
        if not name.startswith('frame_') or extension.lower() not in {'.png', '.jpg', '.jpeg'}:
            continue

        suffix = name.split('_')[-1]
        if not suffix.isdigit():
            continue

        sample_path = os.path.join(label_dir, f'sample_{suffix}.npy')
        if os.path.exists(sample_path):
            continue

        frame_path = os.path.join(label_dir, filename)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        processed_frame = preprocess_frame(frame, calibration)
        sample = extractor(processed_frame)
        if sample is None:
            continue

        metadata = None
        if context is not None and context.write_metadata:
            # Backfill não conhece a captura original: registra o contexto atual
            # e marca a origem para não confundir com uma coleta ao vivo.
            metadata = _build_metadata(label, MODALITY_STATIC, context)
            metadata.source_sample = filename

        generated_samples += _persist_sample(
            label_dir,
            f'sample_{suffix}',
            np.asarray(sample, dtype=np.float32),
            metadata,
            save_mirrored=save_mirrored,
            skip_existing_mirror=True,
        )

    return generated_samples


def _build_hands():
    if not MEDIAPIPE_AVAILABLE:
        raise RuntimeError(
            'MediaPipe não disponível para coleta. '
            f'Motivo: {type(MEDIAPIPE_ERROR).__name__}: {MEDIAPIPE_ERROR}'
        )

    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=COLLECTION_CONFIG['min_detection_confidence'],
        min_tracking_confidence=COLLECTION_CONFIG['min_tracking_confidence'],
    )


def collect_static(
    labels: List[str],
    samples_per_label: int,
    output_dir: str,
    camera_index: int,
    context: Optional[CaptureContext] = None,
    only_missing: bool = True,
):
    context = context or DEFAULT_CAPTURE_CONTEXT

    # Gravar de novo o que já está em disco (coletado aqui ou ingerido de uma
    # base externa) não melhora o modelo e custa a sessão inteira da pessoa.
    labels = resolve_labels_to_collect(
        output_dir, labels, samples_per_label, only_missing, MODALITY_STATIC
    )
    if not labels:
        return

    calibration = _load_optional_calibration()
    hands = _build_hands()
    draw_utils = mp.solutions.drawing_utils
    draw_styles = mp.solutions.drawing_styles
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f'Não foi possível abrir a câmera {camera_index}')

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)

    try:
        for label in labels:
            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            generated_samples = _backfill_static_samples_from_frames(
                label_dir,
                calibration,
                lambda frame: _extract_valid_sample_from_frame(frame, hands),
                save_mirrored=True,
                context=context,
            )
            if generated_samples:
                print(
                    f'[collect-static] {label}: gerados {generated_samples} sample_XXX.npy '
                    'a partir de frames existentes.'
                )

            sample_index = _next_sample_index(label_dir, 'sample_')
            if generated_samples:
                _write_manifest('static', labels, output_dir, samples_per_label, None)

            while sample_index < samples_per_label:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = preprocess_frame(frame, calibration)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                sample = _extract_valid_sample(results)
                preview = frame.copy()

                if results.multi_hand_landmarks:
                    draw_utils.draw_landmarks(
                        preview,
                        results.multi_hand_landmarks[0],
                        mp.solutions.hands.HAND_CONNECTIONS,
                        draw_styles.get_default_hand_landmarks_style(),
                        draw_styles.get_default_hand_connections_style(),
                    )

                _draw_status_overlay(
                    preview,
                    [
                        f'Estatico {label}: {sample_index}/{samples_per_label}',
                        'ESPACO/ENTER captura se landmarks estiverem validos',
                        'Q/ESC sai da coleta',
                    ],
                    accent_color=(0, 255, 0) if sample is not None else (0, 0, 255),
                )
                cv2.imshow(WINDOW_NAME, preview)

                key = cv2.waitKey(1) & 0xFF
                if key in QUIT_KEYS:
                    raise KeyboardInterrupt
                if key in CAPTURE_KEYS and sample is not None:
                    frame_path = os.path.join(
                        label_dir,
                        f'frame_{sample_index:03d}{COLLECTION_CONFIG["static_frame_ext"]}',
                    )
                    metadata = None
                    if context.write_metadata:
                        metadata = _build_metadata(
                            label,
                            MODALITY_STATIC,
                            context,
                            capture_hand=_handedness_label(results),
                            quality=_handedness_score(results),
                        )

                    _persist_sample(
                        label_dir,
                        f'sample_{sample_index:03d}',
                        np.asarray(sample, dtype=np.float32),
                        metadata,
                    )
                    cv2.imwrite(frame_path, frame)
                    sample_index += 1
                    _write_manifest('static', labels, output_dir, samples_per_label, None)

                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    raise KeyboardInterrupt
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


def _wait_for_sequence_start(cap, calibration, label: str, sequence_index: int, total_sequences: int, hands, draw_utils, draw_styles):
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = preprocess_frame(frame, calibration)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        preview = frame.copy()

        if results.multi_hand_landmarks:
            draw_utils.draw_landmarks(
                preview,
                results.multi_hand_landmarks[0],
                mp.solutions.hands.HAND_CONNECTIONS,
                draw_styles.get_default_hand_landmarks_style(),
                draw_styles.get_default_hand_connections_style(),
            )

        _draw_status_overlay(
            preview,
            [
                f'Temporal {label}: sequencia {sequence_index + 1}/{total_sequences}',
                'ESPACO/ENTER inicia a gravacao',
                'Q/ESC sai da coleta',
            ],
        )
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(1) & 0xFF
        if key in QUIT_KEYS:
            raise KeyboardInterrupt
        if key in CAPTURE_KEYS:
            return

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            raise KeyboardInterrupt


def collect_temporal(
    labels: List[str],
    num_sequences: int,
    seq_length: int,
    output_dir: str,
    camera_index: int,
    context: Optional[CaptureContext] = None,
    only_missing: bool = True,
):
    context = context or DEFAULT_CAPTURE_CONTEXT

    labels = resolve_labels_to_collect(
        output_dir, labels, num_sequences, only_missing, MODALITY_TEMPORAL
    )
    if not labels:
        return

    calibration = _load_optional_calibration()
    hands = _build_hands()
    draw_utils = mp.solutions.drawing_utils
    draw_styles = mp.solutions.drawing_styles
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f'Não foi possível abrir a câmera {camera_index}')

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)

    try:
        for label in labels:
            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            sequence_index = _next_sample_index(label_dir, 'seq_')

            while sequence_index < num_sequences:
                _wait_for_sequence_start(
                    cap,
                    calibration,
                    label,
                    sequence_index,
                    num_sequences,
                    hands,
                    draw_utils,
                    draw_styles,
                )

                sequence_frames = []
                frame_scores = []
                capture_hand = None
                recording_started_at = time.monotonic()
                while len(sequence_frames) < seq_length:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    frame = preprocess_frame(frame, calibration)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb_frame)
                    sample = _extract_valid_sample(results)
                    preview = frame.copy()

                    if results.multi_hand_landmarks:
                        draw_utils.draw_landmarks(
                            preview,
                            results.multi_hand_landmarks[0],
                            mp.solutions.hands.HAND_CONNECTIONS,
                            draw_styles.get_default_hand_landmarks_style(),
                            draw_styles.get_default_hand_connections_style(),
                        )

                    if sample is not None:
                        sequence_frames.append(sample)
                        score = _handedness_score(results)
                        if score is not None:
                            frame_scores.append(score)
                        capture_hand = _handedness_label(results) or capture_hand

                    _draw_status_overlay(
                        preview,
                        [
                            f'Temporal {label}: sequencia {sequence_index + 1}/{num_sequences}',
                            f'Gravando frames validos: {len(sequence_frames)}/{seq_length}',
                            'Q/ESC interrompe a coleta',
                        ],
                        accent_color=(0, 255, 0) if sample is not None else (0, 0, 255),
                    )
                    cv2.imshow(WINDOW_NAME, preview)

                    key = cv2.waitKey(1) & 0xFF
                    if key in QUIT_KEYS:
                        raise KeyboardInterrupt

                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        raise KeyboardInterrupt

                metadata = None
                if context.write_metadata:
                    metadata = _build_metadata(
                        label,
                        MODALITY_TEMPORAL,
                        context,
                        capture_hand=capture_hand,
                        quality=float(np.mean(frame_scores)) if frame_scores else None,
                        duration_seconds=time.monotonic() - recording_started_at,
                        sequence_length=seq_length,
                    )

                _persist_sample(
                    label_dir,
                    f'seq_{sequence_index:03d}',
                    np.asarray(sequence_frames, dtype=np.float32),
                    metadata,
                )
                sequence_index += 1
                _write_manifest('temporal', labels, output_dir, num_sequences, seq_length)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


def main():
    parser = argparse.ArgumentParser(description='Coleta unificada do dataset LibrIA')
    parser.add_argument('mode', choices=['static', 'temporal', 'all'], help='Modo de coleta')
    parser.add_argument('--labels', nargs='+', help='Labels a serem coletadas')
    parser.add_argument('--samples-per-label', type=int, default=COLLECTION_CONFIG['static_samples_per_label'])
    parser.add_argument('--num-sequences', type=int, default=COLLECTION_CONFIG['temporal_samples_per_label'])
    parser.add_argument('--seq-length', type=int, default=LSTM_CONFIG['sequence_length'])
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument(
        '--vocabulary',
        choices=VOCABULARY_CHOICES,
        default='alphabet',
        help='Qual parte do vocabulário coletar quando --labels não é informado',
    )
    parser.add_argument('--subject', default=COLLECTION_CONFIG['default_subject_id'],
                        help='Identificador da pessoa que está sinalizando')
    parser.add_argument('--camera-id', default=COLLECTION_CONFIG['default_camera_id'],
                        help='Identificador da câmera usada (modelo/apelido)')
    parser.add_argument('--environment', default=COLLECTION_CONFIG['default_environment'],
                        help='Ambiente da captura (ex.: sala_luz_natural)')
    parser.add_argument('--dominant-hand', default=COLLECTION_CONFIG['default_dominant_hand'],
                        choices=['left', 'right', 'desconhecido'],
                        help='Mão dominante da pessoa')
    parser.add_argument('--no-metadata', action='store_true',
                        help='Não gravar o JSON de metadados junto das amostras')
    parser.add_argument('--all-labels', action='store_true',
                        help='Coleta todas as labels, inclusive as que já atingiram a meta')
    args = parser.parse_args()

    context = CaptureContext(
        subject_id=args.subject,
        camera_id=args.camera_id,
        environment=args.environment,
        dominant_hand=args.dominant_hand,
        write_metadata=not args.no_metadata,
    )

    if context.write_metadata and args.subject == COLLECTION_CONFIG['default_subject_id']:
        print(
            '[coleta] Aviso: --subject não informado. Sem identificar a pessoa não é '
            'possível dividir treino/validação/teste sem vazamento.'
        )

    if args.mode in {'static', 'all'}:
        labels = [
            label.upper()
            for label in (args.labels or _resolve_vocabulary_labels(args.vocabulary, MODALITY_STATIC))
        ]
        collect_static(
            labels,
            args.samples_per_label,
            STATIC_DATASET_DIR,
            args.camera_index,
            context,
            only_missing=not args.all_labels,
        )

    if args.mode in {'temporal', 'all'}:
        labels = [
            label.upper()
            for label in (args.labels or _resolve_vocabulary_labels(args.vocabulary, MODALITY_TEMPORAL))
        ]
        collect_temporal(
            labels,
            args.num_sequences,
            args.seq_length,
            TEMPORAL_DATASET_DIR,
            args.camera_index,
            context,
            only_missing=not args.all_labels,
        )


if __name__ == '__main__':
    main()
