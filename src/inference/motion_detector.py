"""
Detecção de movimento sobre landmarks
=====================================

A energia de movimento é a distância média entre os landmarks do quadro atual
e do anterior, suavizada por média móvel exponencial. É ela que separa "a mão
está parada formando uma letra" de "a mão está executando um sinal".

Trabalhar sobre landmarks, e não sobre pixels, deixa a medida imune a mudança
de fundo e de iluminação — só a mão move a agulha.
"""

from typing import Optional

import numpy as np


class MotionDetector:
    """Energia de movimento suavizada, quadro a quadro."""

    def __init__(self, smoothing: float = 0.5, coordinates_per_point: int = 3):
        if not 0.0 <= smoothing < 1.0:
            raise ValueError('smoothing deve estar em [0.0, 1.0)')
        if coordinates_per_point <= 0:
            raise ValueError('coordinates_per_point deve ser maior que zero')

        self.smoothing = smoothing
        self.coordinates_per_point = coordinates_per_point
        self._previous: Optional[np.ndarray] = None
        self._energy = 0.0
        self._consecutive_frames = 0

    @property
    def energy(self) -> float:
        return self._energy

    @property
    def has_reading(self) -> bool:
        """Indica se a energia atual é uma medida real, e não o zero inicial.

        Um único quadro não tem movimento por definição: só a partir do segundo
        quadro consecutivo com a mão presente o valor significa algo.
        """
        return self._consecutive_frames >= 2

    def reset(self) -> None:
        self._previous = None
        self._energy = 0.0
        self._consecutive_frames = 0

    def _raw_energy(self, features: np.ndarray) -> float:
        if self._previous is None or self._previous.shape != features.shape:
            return 0.0

        delta = features - self._previous
        if delta.size % self.coordinates_per_point == 0:
            # Deslocamento euclidiano médio por ponto, e não média das
            # diferenças por eixo: movimentos diagonais contam por inteiro.
            per_point = delta.reshape(-1, self.coordinates_per_point)
            return float(np.mean(np.linalg.norm(per_point, axis=1)))

        return float(np.mean(np.abs(delta)))

    def update(self, features: Optional[np.ndarray]) -> float:
        """Atualiza a energia com o quadro atual e devolve o valor suavizado.

        ``features=None`` significa mão ausente: a energia decai em vez de
        gerar um pico artificial quando a mão reaparece em outra posição.
        """
        if features is None:
            self._previous = None
            self._consecutive_frames = 0
            self._energy *= self.smoothing
            return self._energy

        current = np.asarray(features, dtype=np.float32).reshape(-1)
        raw_energy = self._raw_energy(current)
        self._previous = current
        self._consecutive_frames += 1

        self._energy = self.smoothing * self._energy + (1.0 - self.smoothing) * raw_energy
        return self._energy
