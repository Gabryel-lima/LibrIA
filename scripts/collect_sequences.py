#!/usr/bin/env python3
"""Coleta sequências temporais de landmarks para treino da LSTM."""

import argparse
import os
import time

import cv2
import numpy as np

from config.settings import CAMERA_CONFIG, FEATURE_DIMENSION, FEATURE_MODE, LSTM_CONFIG, SEQUENCES_DIR
from utils.helpers import extract_landmarks_by_mode, load_camera_calibration, preprocess_frame

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_ERROR = e


WINDOW_NAME = 'LibrIA - Coleta de Sequências'
START_KEYS = {10, 13, 32}
QUIT_KEYS = {27, ord('q')}


def _draw_status_overlay(preview, lines, accent_color=(0, 255, 0)):
    """Desenha informações de status sobre o frame exibido."""
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


def _show_wait_screen(cap, calibration, label, seq_idx, num_sequences, hands, draw_utils, draw_styles):
    """Mantém a janela responsiva enquanto espera o comando para iniciar a gravação."""
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
                f'Label: {label} | Sequência {seq_idx + 1}/{num_sequences}',
                'Pressione ESPACO ou ENTER para iniciar',
                'Pressione Q ou ESC para sair',
            ],
        )
        cv2.imshow(WINDOW_NAME, preview)

        key = cv2.waitKey(1) & 0xFF
        if key in QUIT_KEYS:
            raise KeyboardInterrupt
        if key in START_KEYS:
            return

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            raise KeyboardInterrupt


def _show_countdown(cap, calibration, label, seq_idx, num_sequences, countdown_seconds=3):
    """Exibe uma contagem regressiva curta antes da captura."""
    deadline = time.monotonic() + countdown_seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = preprocess_frame(frame, calibration)
        preview = frame.copy()
        remaining = max(0, int(np.ceil(deadline - time.monotonic())))

        _draw_status_overlay(
            preview,
            [
                f'Label: {label} | Sequência {seq_idx + 1}/{num_sequences}',
                f'Começando em {remaining}...',
                'Mantenha a mão enquadrada',
            ],
            accent_color=(0, 215, 255),
        )
        cv2.imshow(WINDOW_NAME, preview)

        key = cv2.waitKey(1) & 0xFF
        if key in QUIT_KEYS:
            raise KeyboardInterrupt

        if time.monotonic() >= deadline:
            return

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            raise KeyboardInterrupt


def collect_sequences(labels, num_sequences, seq_length, save_dir, camera_index):
    """Grava sequências de landmarks por classe."""
    if not MEDIAPIPE_AVAILABLE:
        raise RuntimeError(
            "MediaPipe não disponível para coleta de sequências. "
            f"Motivo: {type(MEDIAPIPE_ERROR).__name__}: {MEDIAPIPE_ERROR}"
        )

    calibration = None
    if CAMERA_CONFIG['enabled']:
        calibration = load_camera_calibration(
            CAMERA_CONFIG['camera_matrix_path'],
            CAMERA_CONFIG['dist_coeffs_path'],
        )

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.5,
    )
    draw_utils = mp.solutions.drawing_utils
    draw_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a câmera {camera_index}")

    if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(WINDOW_NAME, 800, 600)

    try:
        for label in labels:
            label_dir = os.path.join(save_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            for seq_idx in range(num_sequences):
                sequence = []
                print(f"[{label}] Sequência {seq_idx + 1}/{num_sequences}")
                print('Use a janela da câmera: ESPAÇO/ENTER inicia, Q/ESC cancela.')

                _show_wait_screen(
                    cap,
                    calibration,
                    label,
                    seq_idx,
                    num_sequences,
                    hands,
                    draw_utils,
                    draw_styles,
                )
                _show_countdown(cap, calibration, label, seq_idx, num_sequences)

                while len(sequence) < seq_length:
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
                        features = extract_landmarks_by_mode(
                            results.multi_hand_landmarks[0].landmark,
                            FEATURE_MODE,
                        )
                    else:
                        features = np.zeros(FEATURE_DIMENSION, dtype=np.float32)

                    sequence.append(features)

                    _draw_status_overlay(
                        preview,
                        [
                            f'Label: {label} | Sequência {seq_idx + 1}/{num_sequences}',
                            f'Gravando: {len(sequence)}/{seq_length} frames',
                            'Pressione Q ou ESC para interromper',
                        ],
                    )
                    cv2.imshow(WINDOW_NAME, preview)

                    key = cv2.waitKey(1) & 0xFF
                    if key in QUIT_KEYS:
                        raise KeyboardInterrupt

                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        raise KeyboardInterrupt

                output_path = os.path.join(label_dir, f'seq_{seq_idx:03d}.npy')
                np.save(output_path, np.asarray(sequence, dtype=np.float32))
                print(f"Sequência salva em: {output_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


def main():
    parser = argparse.ArgumentParser(description='Coleta sequências temporais para o LibrIA')
    parser.add_argument('labels', nargs='+', help='Labels a serem coletadas, ex: J Z')
    parser.add_argument('--num-sequences', type=int, default=30, help='Número de sequências por label')
    parser.add_argument(
        '--seq-length',
        type=int,
        default=LSTM_CONFIG['sequence_length'],
        help='Número de frames por sequência',
    )
    parser.add_argument('--save-dir', default=SEQUENCES_DIR, help='Diretório de saída para as sequências')
    parser.add_argument('--camera-index', type=int, default=0, help='Índice da câmera a ser usada')
    args = parser.parse_args()

    collect_sequences(
        labels=args.labels,
        num_sequences=args.num_sequences,
        seq_length=args.seq_length,
        save_dir=args.save_dir,
        camera_index=args.camera_index,
    )


if __name__ == '__main__':
    main()