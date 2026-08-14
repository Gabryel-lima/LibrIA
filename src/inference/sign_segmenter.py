"""
Detecção de início e fim de sinal
=================================

Máquina de estados que transforma um fluxo contínuo de quadros em segmentos
delimitados — o que a janela fixa não consegue fazer, porque sinais têm
durações diferentes e nem todo quadro pertence a um sinal.

Usa histerese: começar exige energia acima de ``start_threshold``, terminar
exige energia abaixo de ``end_threshold`` (menor). Sem essa diferença, um sinal
com uma pausa breve no meio seria cortado em dois.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

STATE_IDLE = 'idle'
STATE_ACTIVE = 'active'

REASON_MOTION_STOPPED = 'motion_stopped'
REASON_MAX_DURATION = 'max_duration'
REASON_HAND_LOST = 'hand_lost'


@dataclass(frozen=True)
class SignSegment:
    """Um trecho delimitado, pronto para virar entrada do modelo temporal."""

    frames: np.ndarray
    start_time: float
    end_time: float
    reason: str

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])

    @property
    def duration_seconds(self) -> float:
        return max(0.0, float(self.end_time) - float(self.start_time))


class SignSegmenter:
    """Delimita sinais a partir da energia de movimento."""

    def __init__(
        self,
        start_threshold: float,
        end_threshold: float,
        min_start_frames: int = 2,
        min_end_frames: int = 5,
        min_frames: int = 4,
        min_duration_seconds: float = 0.15,
        max_duration_seconds: float = 4.0,
        max_absent_frames: int = 5,
    ):
        if end_threshold > start_threshold:
            raise ValueError(
                'end_threshold deve ser menor ou igual a start_threshold (histerese)'
            )
        if min_start_frames <= 0 or min_end_frames <= 0:
            raise ValueError('min_start_frames e min_end_frames devem ser maiores que zero')

        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.min_start_frames = min_start_frames
        self.min_end_frames = min_end_frames
        self.min_frames = min_frames
        self.min_duration_seconds = min_duration_seconds
        self.max_duration_seconds = max_duration_seconds
        self.max_absent_frames = max_absent_frames

        self.state = STATE_IDLE
        self._frames: List[np.ndarray] = []
        self._timestamps: List[float] = []
        self._pending_frames: List[np.ndarray] = []
        self._pending_timestamps: List[float] = []
        self._quiet_streak = 0
        self._absent_streak = 0

    @property
    def is_active(self) -> bool:
        return self.state == STATE_ACTIVE

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def start_time(self) -> Optional[float]:
        return self._timestamps[0] if self._timestamps else None

    def reset(self) -> None:
        self.state = STATE_IDLE
        self._frames = []
        self._timestamps = []
        self._pending_frames = []
        self._pending_timestamps = []
        self._quiet_streak = 0
        self._absent_streak = 0

    def update(
        self,
        features: Optional[np.ndarray],
        timestamp: float,
        energy: float,
        hand_present: bool = True,
    ) -> Optional[SignSegment]:
        """Processa um quadro e devolve o segmento quando um sinal se encerra."""
        if self.state == STATE_IDLE:
            return self._update_idle(features, timestamp, energy, hand_present)
        return self._update_active(features, timestamp, energy, hand_present)

    def _update_idle(
        self,
        features: Optional[np.ndarray],
        timestamp: float,
        energy: float,
        hand_present: bool,
    ) -> Optional[SignSegment]:
        if not hand_present or features is None or energy < self.start_threshold:
            self._pending_frames = []
            self._pending_timestamps = []
            return None

        # Acumula desde o primeiro quadro acima do limiar: o começo do sinal
        # não pode ser descartado enquanto confirmamos que é mesmo um sinal.
        self._pending_frames.append(np.asarray(features, dtype=np.float32).reshape(-1))
        self._pending_timestamps.append(float(timestamp))

        if len(self._pending_frames) >= self.min_start_frames:
            self.state = STATE_ACTIVE
            self._frames = self._pending_frames
            self._timestamps = self._pending_timestamps
            self._pending_frames = []
            self._pending_timestamps = []
            self._quiet_streak = 0
            self._absent_streak = 0

        return None

    def _update_active(
        self,
        features: Optional[np.ndarray],
        timestamp: float,
        energy: float,
        hand_present: bool,
    ) -> Optional[SignSegment]:
        if hand_present and features is not None:
            self._frames.append(np.asarray(features, dtype=np.float32).reshape(-1))
            self._timestamps.append(float(timestamp))
            self._absent_streak = 0
        else:
            self._absent_streak += 1
            if self._absent_streak >= self.max_absent_frames:
                return self._close(REASON_HAND_LOST, trim=0)

        if energy < self.end_threshold:
            self._quiet_streak += 1
            if self._quiet_streak >= self.min_end_frames:
                # Os quadros parados do fim não pertencem ao sinal.
                return self._close(REASON_MOTION_STOPPED, trim=self._quiet_streak)
        else:
            self._quiet_streak = 0

        if self._timestamps and (timestamp - self._timestamps[0]) >= self.max_duration_seconds:
            return self._close(REASON_MAX_DURATION, trim=0)

        return None

    def _close(self, reason: str, trim: int) -> Optional[SignSegment]:
        frames = self._frames
        timestamps = self._timestamps

        if trim > 0 and len(frames) - trim >= self.min_frames:
            frames = frames[:-trim]
            timestamps = timestamps[:-trim]

        self.reset()

        if len(frames) < self.min_frames or not timestamps:
            return None

        duration = timestamps[-1] - timestamps[0]
        if duration < self.min_duration_seconds:
            return None

        return SignSegment(
            frames=np.asarray(frames, dtype=np.float32),
            start_time=timestamps[0],
            end_time=timestamps[-1],
            reason=reason,
        )
