#!/usr/bin/env python3
"""Exibe em tela cheia uma imagem de tabuleiro para calibração."""

import argparse
import os

import cv2


def main():
    parser = argparse.ArgumentParser(description='Exibe um tabuleiro de calibração em tela cheia')
    parser.add_argument(
        '--image',
        default='output/checkerboard_9x6.png',
        help='Arquivo de imagem do tabuleiro a ser exibido',
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(
            f'Imagem do tabuleiro não encontrada: {args.image}. Rode o gerador antes de exibir.'
        )

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f'Não foi possível abrir a imagem: {args.image}')

    window_name = 'LibrIA - Tabuleiro de Calibracao'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            preview = image.copy()
            cv2.putText(
                preview,
                'Mostre este tabuleiro para a camera. Pressione q para sair.',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.imshow(window_name, preview)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()