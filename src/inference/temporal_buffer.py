"""
Buffer temporal de landmarks
============================

Substitui a janela fixa de ``deque(maxlen=sequence_length)`` por um buffer que
guarda o histórico com carimbo de tempo e sabe reamostrar um trecho para o
comprimento que o modelo espera.

Isso é o que permite segmentar sinais de duração variável: o segmento tem o
tamanho que o gesto teve, e só na hora de inferir ele é ajustado para
``sequence_length``.
"""

from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np


def resample_sequence(frames: Sequence[np.ndarray], target_length: int) -> np.ndarray:
    """Reamostra ``frames`` para exatamente ``target_length`` passos.

    Usa vizinho mais próximo sobre índices igualmente espaçados: preserva as
    poses do gesto sem inventar quadros interpolados entre posições distantes.
    """
    if target_length <= 0:
        raise ValueError('target_length deve ser maior que zero')

    array = np.asarray(frames, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f'Esperado array 2D (frames, features), recebido shape {array.shape}')
    if array.shape[0] == 0:
        raise ValueError('Não é possível reamostrar uma sequência vazia')

    if array.shape[0] == target_length:
        return array.copy()

    indices = np.linspace(0, array.shape[0] - 1, target_length)
    return array[np.rint(indices).astype(int)]


class TemporalBuffer:
    """Histórico deslizante de features com carimbo de tempo."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError('capacity deve ser maior que zero')

        self.capacity = capacity
        self._frames: Deque[np.ndarray] = deque(maxlen=capacity)
        self._timestamps: Deque[float] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def is_full(self) -> bool:
        return len(self._frames) >= self.capacity

    @property
    def timestamps(self) -> List[float]:
        return list(self._timestamps)

    def append(self, features: np.ndarray, timestamp: float) -> None:
        self._frames.append(np.asarray(features, dtype=np.float32).reshape(-1))
        self._timestamps.append(float(timestamp))

    def reset(self) -> None:
        self._frames.clear()
        self._timestamps.clear()

    def frames(self) -> List[np.ndarray]:
        return list(self._frames)

    def window(self, length: Optional[int] = None) -> Optional[np.ndarray]:
        """Últimos ``length`` quadros como array, ou ``None`` se não houver tantos."""
        size = self.capacity if length is None else length
        if size <= 0 or len(self._frames) < size:
            return None
        return np.asarray(list(self._frames)[-size:], dtype=np.float32)

    def window_span(self, length: Optional[int] = None) -> Optional[Tuple[float, float]]:
        """Intervalo (início, fim) da janela correspondente a :meth:`window`."""
        size = self.capacity if length is None else length
        if size <= 0 or len(self._timestamps) < size:
            return None
        timestamps = list(self._timestamps)[-size:]
        return timestamps[0], timestamps[-1]
