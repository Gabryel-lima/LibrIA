"""Regras de arbitragem para inferência híbrida de Libras."""

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class PredictionEvent:
    """Representa uma predição emitida por um dos modelos."""

    token: str
    confidence: float
    source: str
    start_time: float
    end_time: float
    frame_index: int


class PredictionMerger:
    """Resolve conflitos entre predições estáticas e temporais."""

    def __init__(
        self,
        temporal_priority_classes: Sequence[str],
        temporal_confidence_threshold: float,
        static_confidence_threshold: float,
        cooldown_seconds: float,
        history_size: int = 5,
    ):
        self.temporal_priority_classes = {token.upper() for token in temporal_priority_classes}
        self.temporal_confidence_threshold = temporal_confidence_threshold
        self.static_confidence_threshold = static_confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        self.history_size = history_size
        self.history: List[PredictionEvent] = []
        self.active_temporal_until = 0.0
        self.last_emitted_event: Optional[PredictionEvent] = None

    def _push_history(self, event: PredictionEvent):
        self.history.append(event)
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]

    def submit(self, event: PredictionEvent) -> Optional[PredictionEvent]:
        token_upper = event.token.upper()

        if event.source == 'temporal':
            if token_upper not in self.temporal_priority_classes:
                return None
            if event.confidence < self.temporal_confidence_threshold:
                return None

            self.active_temporal_until = max(self.active_temporal_until, event.end_time + self.cooldown_seconds)
            self.last_emitted_event = event
            self._push_history(event)
            return event

        if event.source == 'static':
            if event.confidence < self.static_confidence_threshold:
                return None
            if event.start_time <= self.active_temporal_until:
                return None

            self.last_emitted_event = event
            self._push_history(event)
            return event

        return None