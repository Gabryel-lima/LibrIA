"""
Pipeline temporal robusto
=========================

Orquestra os componentes da Fase 2 em um único fluxo quadro a quadro:

    features → buffer → movimento → segmentação → modelo → suavização
             → supressão de duplicatas → SignToken

Diferente da janela fixa anterior, que classificava a cada 30 quadros
independentemente de haver ou não um sinal em execução, aqui o modelo temporal
só é consultado quando existe movimento delimitado, e o modelo estático segue
como fallback enquanto a mão está parada formando uma letra.

Os modelos entram por injeção de dependência (``temporal_predictor`` e
``static_predictor``), o que mantém o pipeline testável sem TensorFlow e
permite comparar LSTM, CNN temporal e Transformer sem tocar nesta classe.
"""

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from config.settings import TEMPORAL_PIPELINE_CONFIG
from config.vocabulary import UNKNOWN_LABEL

from .motion_detector import MotionDetector
from .prediction_smoother import DuplicateSuppressor, ProbabilitySmoother
from .sign_segmenter import SignSegment, SignSegmenter
from .sign_token import (
    SOURCE_STATIC,
    SOURCE_TEMPORAL,
    STATE_PARTIAL,
    SignToken,
    build_rejected_token,
    build_token,
)
from .temporal_buffer import TemporalBuffer, resample_sequence

TemporalPredictor = Callable[[np.ndarray], Sequence[float]]
StaticPredictor = Callable[[np.ndarray], Tuple[str, float]]


class TemporalPipeline:
    """Reconhecimento temporal com segmentação, suavização e deduplicação."""

    def __init__(
        self,
        temporal_predictor: TemporalPredictor,
        label_map: Dict[int, str],
        sequence_length: int,
        static_predictor: Optional[StaticPredictor] = None,
        config: Optional[dict] = None,
    ):
        resolved = dict(TEMPORAL_PIPELINE_CONFIG)
        resolved.update(config or {})
        self.config = resolved

        self.temporal_predictor = temporal_predictor
        self.static_predictor = static_predictor
        self.label_map = dict(label_map)
        self.sequence_length = sequence_length

        self.buffer = TemporalBuffer(capacity=sequence_length)
        self.motion = MotionDetector(smoothing=resolved['motion_smoothing'])
        self.segmenter = SignSegmenter(
            start_threshold=resolved['motion_start_threshold'],
            end_threshold=resolved['motion_end_threshold'],
            min_start_frames=resolved['min_start_frames'],
            min_end_frames=resolved['min_end_frames'],
            min_frames=resolved['min_segment_frames'],
            min_duration_seconds=resolved['min_duration_seconds'],
            max_duration_seconds=resolved['max_duration_seconds'],
            max_absent_frames=resolved['max_absent_frames'],
        )
        self.smoother = ProbabilitySmoother(window_size=resolved['smoothing_window'])
        self.suppressor = DuplicateSuppressor(window_seconds=resolved['duplicate_window_seconds'])

        self.frame_index = 0
        self.last_energy = 0.0
        self.last_token: Optional[SignToken] = None
        self.static_blocked_until = 0.0

    @property
    def is_signing(self) -> bool:
        """Indica se há um sinal em execução (para feedback na interface)."""
        return self.segmenter.is_active

    def reset(self) -> None:
        self.buffer.reset()
        self.motion.reset()
        self.segmenter.reset()
        self.smoother.reset()
        self.suppressor.reset()
        self.frame_index = 0
        self.last_energy = 0.0
        self.last_token = None
        self.static_blocked_until = 0.0

    def process_frame(
        self,
        features: Optional[np.ndarray],
        timestamp: float,
        hand_present: bool = True,
    ) -> Optional[SignToken]:
        """Processa um quadro e devolve um token quando houver o que reportar."""
        present = hand_present and features is not None
        self.frame_index += 1
        self.last_energy = self.motion.update(features if present else None)

        if present:
            self.buffer.append(features, timestamp)

        segment = self.segmenter.update(
            features if present else None,
            timestamp,
            self.last_energy,
            hand_present=present,
        )

        if segment is not None:
            return self._emit(self._finalize_segment(segment))

        if self.segmenter.is_active:
            return self._emit(self._partial_token(timestamp))

        return self._emit(self._static_token(features if present else None, timestamp))

    def _emit(self, token: Optional[SignToken]) -> Optional[SignToken]:
        if token is not None:
            self.last_token = token
        return token

    def _predict_sequence(self, sequence: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(self.temporal_predictor(sequence), dtype=np.float64).reshape(-1)
        if probabilities.size == 0:
            raise ValueError('temporal_predictor devolveu uma distribuição vazia')
        return probabilities

    def _label_for(self, index: int) -> str:
        return str(self.label_map.get(index, index)).upper()

    def _partial_token(self, timestamp: float) -> Optional[SignToken]:
        """Hipótese durante a execução do sinal, para feedback imediato."""
        if not self.config['emit_partial_tokens']:
            return None
        if self.frame_index % self.config['partial_interval_frames'] != 0:
            return None

        window = self.buffer.window(self.sequence_length)
        if window is None:
            return None

        self.smoother.update(self._predict_sequence(window))
        best = self.smoother.best()
        if best is None:
            return None

        index, confidence = best
        start_time = self.segmenter.start_time
        return build_token(
            label=self._label_for(index),
            confidence=confidence,
            start_time=timestamp if start_time is None else start_time,
            end_time=timestamp,
            source=SOURCE_TEMPORAL,
            state=STATE_PARTIAL,
            frame_count=self.segmenter.frame_count,
        )

    def _finalize_segment(self, segment: SignSegment) -> Optional[SignToken]:
        """Classifica o segmento fechado e decide entre aceitar e recusar."""
        sequence = resample_sequence(segment.frames, self.sequence_length)
        # A predição do segmento completo entra na mesma janela das parciais:
        # a decisão final considera todo o sinal, não só o último instante.
        self.smoother.update(self._predict_sequence(sequence))
        best = self.smoother.best()
        self.smoother.reset()

        if best is None:
            return None

        index, confidence = best
        label = self._label_for(index)
        rejected = confidence < self.config['temporal_confidence_threshold']
        emitted_label = UNKNOWN_LABEL if rejected else label

        # O retorno da mão ao repouso não pode ser lido como letra.
        self.static_blocked_until = segment.end_time + self.config['static_cooldown_seconds']

        if not self.suppressor.accept(emitted_label, segment.end_time):
            return None

        if rejected:
            return build_rejected_token(
                start_time=segment.start_time,
                end_time=segment.end_time,
                source=SOURCE_TEMPORAL,
                confidence=confidence,
                frame_count=segment.frame_count,
            )

        return build_token(
            label=label,
            confidence=confidence,
            start_time=segment.start_time,
            end_time=segment.end_time,
            source=SOURCE_TEMPORAL,
            frame_count=segment.frame_count,
        )

    def _static_token(
        self,
        features: Optional[np.ndarray],
        timestamp: float,
    ) -> Optional[SignToken]:
        """Fallback estático: a mão parada ainda pode estar formando uma letra."""
        if self.static_predictor is None or features is None:
            return None
        # Letra é mão parada: sem leitura de movimento confiável, ou com a mão
        # ainda em deslocamento, o modelo estático não deve opinar.
        if not self.motion.has_reading:
            return None
        if self.last_energy >= self.config['motion_end_threshold']:
            return None
        if timestamp < self.static_blocked_until:
            return None
        if self.frame_index % self.config['static_interval_frames'] != 0:
            return None

        label, confidence = self.static_predictor(features)
        if confidence < self.config['static_confidence_threshold']:
            return None

        normalized = str(label).strip().upper()
        if not self.suppressor.accept(normalized, timestamp):
            return None

        return build_token(
            label=normalized,
            confidence=confidence,
            start_time=timestamp,
            end_time=timestamp,
            source=SOURCE_STATIC,
            frame_count=1,
        )
