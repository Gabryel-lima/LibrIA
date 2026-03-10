#!/usr/bin/env python3
"""Gera uma imagem de tabuleiro para calibração de câmera."""

import argparse
import os

import cv2
import numpy as np


def generate_checkerboard(cols, rows, square_size, margin):
    """Gera um tabuleiro com base no número de cantos internos desejado."""
    num_square_cols = cols + 1
    num_square_rows = rows + 1

    board_width = num_square_cols * square_size
    board_height = num_square_rows * square_size

    image = np.full(
        (board_height + margin * 2, board_width + margin * 2),
        255,
        dtype=np.uint8,
    )

    for row in range(num_square_rows):
        for col in range(num_square_cols):
            if (row + col) % 2 == 0:
                start_x = margin + col * square_size
                start_y = margin + row * square_size
                end_x = start_x + square_size
                end_y = start_y + square_size
                image[start_y:end_y, start_x:end_x] = 0

    return image


def main():
    parser = argparse.ArgumentParser(description='Gera um tabuleiro para calibração')
    parser.add_argument('--cols', type=int, default=9, help='Número de colunas internas do tabuleiro')
    parser.add_argument('--rows', type=int, default=6, help='Número de linhas internas do tabuleiro')
    parser.add_argument('--square-size', type=int, default=80, help='Tamanho de cada quadrado em pixels')
    parser.add_argument('--margin', type=int, default=40, help='Margem branca ao redor do tabuleiro em pixels')
    parser.add_argument(
        '--output',
        default='output/checkerboard_9x6.png',
        help='Arquivo de saída da imagem gerada',
    )
    args = parser.parse_args()

    image = generate_checkerboard(args.cols, args.rows, args.square_size, args.margin)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    cv2.imwrite(args.output, image)

    print(f'Tabuleiro salvo em: {args.output}')
    print(f'Use o padrão com {args.cols}x{args.rows} cantos internos.')


if __name__ == '__main__':
    main()