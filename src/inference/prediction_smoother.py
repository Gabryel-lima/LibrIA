"""
Suavização de predições e supressão de duplicatas
=================================================

Duas correções independentes sobre a saída crua do modelo:

* :class:`ProbabilitySmoother` — média das últimas distribuições de
  probabilidade. Uma classificação isolada e errada no meio de um sinal deixa
  de trocar o token exibido.
* :class:`DuplicateSuppressor` — o mesmo sinal sustentado por meio segundo
  gera várias predições idênticas; sem isso, "OI" vira "OI OI OI" na tradução.
"""

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np


class ProbabilitySmoother:
    """Média móvel das distribuições de probabilidade recentes."""

    def __init__(self, window_size: int = 5):
        if window_size <= 0:
            raise ValueError('window_size deve ser maior que zero')

        self.window_size = window_size
        self._window: Deque[np.ndarray] = deque(maxlen=window_size)

    def __len__(self) -> int:
        return len(self._window)

    def reset(self) -> None:
        self._window.clear()

    def update(self, probabilities: np.ndarray) -> np.ndarray:
        """Adiciona uma distribuição e devolve a média das que estão na janela."""
        array = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if array.size == 0:
            raise ValueError('probabilities não pode ser vazio')
        if self._window and array.size != self._window[-1].size:
            raise ValueError('Todas as distribuições devem ter o mesmo tamanho')

        self._window.append(array)
        return self.average()

    def average(self) -> Optional[np.ndarray]:
        if not self._window:
            return None
        return np.mean(np.stack(self._window), axis=0)

    def best(self) -> Optional[Tuple[int, float]]:
        """Índice e confiança da classe mais provável na média da janela."""
        averaged = self.average()
        if averaged is None:
            return None
        best_index = int(np.argmax(averaged))
        return best_index, float(averaged[best_index])


class DuplicateSuppressor:
    """Bloqueia repetições do mesmo rótulo dentro de uma janela de tempo."""

    def __init__(self, window_seconds: float = 1.0):
        if window_seconds < 0:
            raise ValueError('window_seconds não pode ser negativo')

        self.window_seconds = window_seconds
        self.last_label: Optional[str] = None
        self.last_timestamp: Optional[float] = None

    def reset(self) -> None:
        self.last_label = None
        self.last_timestamp = None

    def should_emit(self, label: str, timestamp: float) -> bool:
        """Informa se o rótulo pode ser emitido, sem registrar a emissão."""
        if self.last_label is None or self.last_timestamp is None:
            return True
        if label != self.last_label:
            return True
        return (timestamp - self.last_timestamp) >= self.window_seconds

    def register(self, label: str, timestamp: float) -> None:
        self.last_label = label
        self.last_timestamp = float(timestamp)

    def accept(self, label: str, timestamp: float) -> bool:
        """Verifica e registra em uma única chamada."""
        if not self.should_emit(label, timestamp):
            # Um sinal sustentado renova a janela: só volta a emitir depois de
            # window_seconds sem o sinal, não window_seconds após a 1ª emissão.
            self.register(label, timestamp)
            return False

        self.register(label, timestamp)
        return True
