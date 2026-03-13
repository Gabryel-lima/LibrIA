#!/usr/bin/env python3
"""Coleta unificada de dataset estático e temporal para o LibrIA."""

import argparse
import json
import os
import time
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from config.settings import (
    CAMERA_CONFIG,
    COLLECTION_CONFIG,
    FEATURE_DIMENSION,
    FEATURE_MODE,
    LSTM_CONFIG,
    STATIC_DATASET_DIR,
    STATIC_LABELS,
    TEMPORAL_DATASET_DIR,
    TEMPORAL_LABELS,
)
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


def _storage_shape(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if FEATURE_MODE == 'wrist_relative' and array.size == FEATURE_DIMENSION and FEATURE_DIMENSION % 3 == 0:
        return array.reshape(-1, 3)
    return array


def _next_sample_index(label_dir: str, prefix: str) -> int:
    existing_indices = []
    for filename in os.listdir(label_dir):
        if not filename.startswith(prefix) or not filename.endswith('.npy'):
            continue
        stem = os.path.splitext(filename)[0]
        suffix = stem.split('_')[-1]
        if suffix.isdigit():
            existing_indices.append(int(suffix))
    return max(existing_indices, default=-1) + 1


def _write_manifest(mode: str, labels: List[str], output_dir: str, sample_target: int, sequence_length: Optional[int]):
    manifest_path = os.path.join(output_dir, 'manifest.json')
    samples = {}
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
        'camera_calibrated': _load_optional_calibration() is not None,
        'labels': labels,
        'counts': samples,
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(manifest_path, 'w', encoding='utf-8') as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=True)


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
) -> int:
    generated_samples = 0

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

        np.save(sample_path, np.asarray(sample, dtype=np.float32))
        generated_samples += 1

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


def collect_static(labels: List[str], samples_per_label: int, output_dir: str, camera_index: int):
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
                    sample_path = os.path.join(label_dir, f'sample_{sample_index:03d}.npy')
                    frame_path = os.path.join(
                        label_dir,
                        f'frame_{sample_index:03d}{COLLECTION_CONFIG["static_frame_ext"]}',
                    )
                    np.save(sample_path, np.asarray(sample, dtype=np.float32))
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


def collect_temporal(labels: List[str], num_sequences: int, seq_length: int, output_dir: str, camera_index: int):
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

                sequence_path = os.path.join(label_dir, f'seq_{sequence_index:03d}.npy')
                np.save(sequence_path, np.asarray(sequence_frames, dtype=np.float32))
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
    args = parser.parse_args()

    if args.mode in {'static', 'all'}:
        labels = [label.upper() for label in (args.labels or STATIC_LABELS)]
        collect_static(labels, args.samples_per_label, STATIC_DATASET_DIR, args.camera_index)

    if args.mode in {'temporal', 'all'}:
        labels = [label.upper() for label in (args.labels or TEMPORAL_LABELS)]
        collect_temporal(labels, args.num_sequences, args.seq_length, TEMPORAL_DATASET_DIR, args.camera_index)


if __name__ == '__main__':
    main()