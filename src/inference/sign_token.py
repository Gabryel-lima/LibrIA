"""
Saída padronizada do reconhecimento temporal
============================================

Todo reconhecimento — estático ou temporal, parcial ou final — sai como um
:class:`SignToken`. É o contrato entre a camada de reconhecimento (Fase 2) e a
camada de composição linguística (Fases 3 e 4): quem consome tokens não
precisa saber qual modelo os produziu.

O ``state`` é o que evita apresentar tradução errada como certeza:

* ``partial``  - hipótese enquanto o sinal ainda está sendo executado;
* ``final``    - sinal encerrado e aceito acima do limiar de confiança;
* ``rejected`` - sinal encerrado mas abaixo do limiar (fora do vocabulário).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from config.vocabulary import UNKNOWN_LABEL, get_sign_type, get_translation

STATE_PARTIAL = 'partial'
STATE_FINAL = 'final'
STATE_REJECTED = 'rejected'
TOKEN_STATES = (STATE_PARTIAL, STATE_FINAL, STATE_REJECTED)

SOURCE_STATIC = 'static'
SOURCE_TEMPORAL = 'temporal'

UNSPECIFIED_SIGN_TYPE = 'desconhecido'


@dataclass(frozen=True)
class SignToken:
    """Uma unidade reconhecida, com janela temporal e estado de finalização."""

    label: str
    confidence: float
    start_time: float
    end_time: float
    source: str
    state: str = STATE_FINAL
    sign_type: str = UNSPECIFIED_SIGN_TYPE
    frame_count: int = 0

    @property
    def duration_seconds(self) -> float:
        return max(0.0, float(self.end_time) - float(self.start_time))

    @property
    def is_final(self) -> bool:
        return self.state == STATE_FINAL

    @property
    def is_rejected(self) -> bool:
        return self.state == STATE_REJECTED

    @property
    def token(self) -> str:
        """Texto a compor. Vazio para gestos funcionais e para rejeições."""
        if self.state == STATE_REJECTED:
            return ''
        return get_translation(self.label) or ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'label': self.label,
            'token': self.token,
            'confidence': self.confidence,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_seconds': self.duration_seconds,
            'source': self.source,
            'state': self.state,
            'sign_type': self.sign_type,
            'frame_count': self.frame_count,
        }


def build_token(
    label: str,
    confidence: float,
    start_time: float,
    end_time: float,
    source: str,
    state: str = STATE_FINAL,
    frame_count: int = 0,
    sign_type: Optional[str] = None,
) -> SignToken:
    """Cria um :class:`SignToken` resolvendo o tipo de sinal pelo vocabulário."""
    if state not in TOKEN_STATES:
        raise ValueError(f'Estado inválido para token: {state}')

    normalized_label = str(label).strip().upper()
    resolved_type = sign_type or get_sign_type(normalized_label) or UNSPECIFIED_SIGN_TYPE

    return SignToken(
        label=normalized_label,
        confidence=float(confidence),
        start_time=float(start_time),
        end_time=float(end_time),
        source=source,
        state=state,
        sign_type=resolved_type,
        frame_count=int(frame_count),
    )


def build_rejected_token(
    start_time: float,
    end_time: float,
    source: str,
    confidence: float = 0.0,
    frame_count: int = 0,
) -> SignToken:
    """Token de recusa: o sinal terminou, mas não há confiança para afirmá-lo."""
    return build_token(
        label=UNKNOWN_LABEL,
        confidence=confidence,
        start_time=start_time,
        end_time=end_time,
        source=source,
        state=STATE_REJECTED,
        frame_count=frame_count,
    )
