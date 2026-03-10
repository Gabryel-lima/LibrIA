#!/usr/bin/env python3
"""Script para calibrar a câmera usada na inferência."""

import argparse
import glob
import os
import time

import cv2
import numpy as np

from config.settings import CAMERA_CONFIG


def expand_image_patterns(patterns):
    """Expande globs recebidos na linha de comando."""
    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob.glob(pattern))
    return sorted(set(image_paths))


def capture_calibration_images(chessboard_size, capture_dir, target_images, camera_index):
    """Abre a webcam, detecta o tabuleiro e salva imagens válidas para calibração."""
    os.makedirs(capture_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a câmera {camera_index}")

    saved_paths = []
    cols, rows = chessboard_size
    window_name = 'LibrIA - Captura de Calibracao'

    print('Captura de calibração iniciada.')
    print(f'Use um tabuleiro com {cols}x{rows} cantos internos.')
    print("Controles: 'espaco' salva uma imagem válida, 'q' encerra a captura.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            preview = frame.copy()
            status_text = 'Tabuleiro detectado' if found else 'Ajuste o tabuleiro no enquadramento'
            status_color = (0, 200, 0) if found else (0, 0, 255)

            if found:
                cv2.drawChessboardCorners(preview, chessboard_size, corners, found)

            cv2.putText(
                preview,
                f'Padrao: {cols}x{rows} | Salvas: {len(saved_paths)}/{target_images}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                preview,
                status_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )
            cv2.putText(
                preview,
                "Espaco = salvar | q = sair",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                2,
            )
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if key == ord(' ') and found:
                filename = f'calibration_{len(saved_paths):02d}_{int(time.time() * 1000)}.jpg'
                output_path = os.path.join(capture_dir, filename)
                cv2.imwrite(output_path, frame)
                saved_paths.append(output_path)
                print(f'Imagem salva: {output_path}')

                if len(saved_paths) >= target_images:
                    print('Quantidade alvo de imagens atingida.')
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return saved_paths


def calibrate_camera(image_paths, chessboard_size):
    """Calcula a matriz da câmera e coeficientes de distorção."""
    objpoints = []
    imgpoints = []

    cols, rows = chessboard_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    gray_shape = None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ignorando imagem inválida: {image_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if not found:
            print(f"Tabuleiro não detectado: {image_path}")
            continue

        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(refined_corners)
        gray_shape = gray.shape[::-1]
        print(f"Tabuleiro detectado: {image_path}")

    if not objpoints or gray_shape is None:
        raise RuntimeError("Nenhuma imagem válida com tabuleiro detectado foi encontrada")

    _, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray_shape,
        None,
        None,
    )
    return camera_matrix, dist_coeffs


def main():
    parser = argparse.ArgumentParser(description='Calibra a câmera do LibrIA')
    parser.add_argument(
        'images',
        nargs='*',
        help='Arquivos ou globs com imagens do tabuleiro, ex: calibration/*.jpg',
    )
    parser.add_argument('--cols', type=int, default=9, help='Número de colunas internas do tabuleiro')
    parser.add_argument('--rows', type=int, default=6, help='Número de linhas internas do tabuleiro')
    parser.add_argument(
        '--capture',
        action='store_true',
        help='Abre a câmera, detecta o tabuleiro e salva imagens antes de calibrar',
    )
    parser.add_argument(
        '--capture-dir',
        default='calibration',
        help='Diretório para salvar imagens capturadas de calibração',
    )
    parser.add_argument(
        '--target-images',
        type=int,
        default=15,
        help='Quantidade alvo de imagens válidas para captura de calibração',
    )
    parser.add_argument(
        '--camera-index',
        type=int,
        default=0,
        help='Índice da câmera usada no modo de captura',
    )
    parser.add_argument(
        '--camera-matrix-path',
        default=CAMERA_CONFIG['camera_matrix_path'],
        help='Arquivo de saída para a matriz da câmera',
    )
    parser.add_argument(
        '--dist-coeffs-path',
        default=CAMERA_CONFIG['dist_coeffs_path'],
        help='Arquivo de saída para os coeficientes de distorção',
    )
    args = parser.parse_args()

    image_paths = expand_image_patterns(args.images)
    if args.capture:
        image_paths = capture_calibration_images(
            chessboard_size=(args.cols, args.rows),
            capture_dir=args.capture_dir,
            target_images=args.target_images,
            camera_index=args.camera_index,
        )

    if not image_paths:
        raise FileNotFoundError(
            'Nenhuma imagem encontrada para calibração. Use imagens de um tabuleiro '
            'ou rode com --capture para coletá-las via webcam.'
        )

    camera_matrix, dist_coeffs = calibrate_camera(image_paths, (args.cols, args.rows))

    output_dir = os.path.dirname(args.camera_matrix_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    np.save(args.camera_matrix_path, camera_matrix)
    np.save(args.dist_coeffs_path, dist_coeffs)

    print(f"Matriz da câmera salva em: {args.camera_matrix_path}")
    print(f"Coeficientes de distorção salvos em: {args.dist_coeffs_path}")


if __name__ == '__main__':
    main()