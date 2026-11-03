"""Escolha de qual mão classificar quando o MediaPipe detecta mais de uma.

Os classificadores do LibrIA recebem **uma** mão (63 ou 42 features). Até aqui
o MediaPipe rodava com ``max_num_hands=1`` e o código lia
``results.multi_hand_landmarks[0]``. Isso tem dois problemas:

1. Com ``max_num_hands=1`` o detector devolve a primeira mão que achar, não a
   que está sinalizando. Se a outra mão entra no quadro (apoiada, coçando o
   rosto, de outra pessoa), ela pode "roubar" a detecção.
2. A ordem de ``multi_hand_landmarks`` não é estável entre quadros. Numa
   sequência temporal, trocar de mão no meio do sinal corrompe a janela inteira.

Este módulo resolve a escolha em duas etapas:

* **Estratégia** (``first``, ``largest``, ``most_confident``, ``dominant``):
  decide qual mão escolher quando não há histórico.
* **Rastreamento**: com uma mão já escolhida no quadro anterior, mantém a
  candidata cujo punho está mais perto do punho anterior (se o salto for
  menor que ``tracking_max_jump``). É uma histerese espacial: impede que duas
  mãos de tamanho parecido alternem a cada quadro.

Lateralidade: o MediaPipe rotula ``Left``/``Right`` **supondo imagem
espelhada** (selfie). O LibrIA não espelha os quadros da webcam nem dos
vídeos externos, então o rótulo bruto vem invertido em relação à mão real da
pessoa. ``person_hand`` já devolve a mão real quando
``image_is_mirrored=False``.

Não há suporte a sinais feitos com as duas mãos: isso exige mudar o contrato
de features dos modelos (2 × 63) e está registrado no ``ROADMAP.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

STRATEGY_FIRST = 'first'
STRATEGY_LARGEST = 'largest'
STRATEGY_MOST_CONFIDENT = 'most_confident'
STRATEGY_DOMINANT = 'dominant'

STRATEGIES = (
    STRATEGY_FIRST,
    STRATEGY_LARGEST,
    STRATEGY_MOST_CONFIDENT,
    STRATEGY_DOMINANT,
)

HAND_LEFT = 'left'
HAND_RIGHT = 'right'

WRIST_INDEX = 0

DEFAULT_CONFIG: Dict[str, Any] = {
    'max_num_hands': 2,
    'strategy': STRATEGY_LARGEST,
    # Mão dominante real da pessoa ('left'/'right'); usada só por 'dominant'.
    'dominant_hand': None,
    # True apenas se o quadro já chega espelhado (câmera selfie com flip).
    'image_is_mirrored': False,
    # Salto máximo do punho entre quadros (coordenadas normalizadas 0..1)
    # para considerar que é a mesma mão.
    'tracking_max_jump': 0.15,
}


def _normalize_hand(hand: Optional[str]) -> Optional[str]:
    if hand is None:
        return None
    normalized = str(hand).strip().lower()
    if normalized in {'l', 'left', 'esquerda', 'e'}:
        return HAND_LEFT
    if normalized in {'r', 'right', 'direita', 'd'}:
        return HAND_RIGHT
    return None


def _swap_hand(hand: Optional[str]) -> Optional[str]:
    if hand == HAND_LEFT:
        return HAND_RIGHT
    if hand == HAND_RIGHT:
        return HAND_LEFT
    return None


@dataclass(frozen=True)
class HandCandidate:
    """Uma mão detectada no quadro, com o que é preciso para escolhê-la."""

    index: int
    hand_landmarks: Any
    score: Optional[float]
    mediapipe_label: Optional[str]
    person_hand: Optional[str]
    bbox_area: float
    wrist: tuple

    @property
    def landmarks(self) -> Sequence[Any]:
        """Lista de 21 pontos (``.x``, ``.y``, ``.z``), como o extrator espera."""
        return self.hand_landmarks.landmark


def _bbox_area(points: Sequence[Any]) -> float:
    xs = [float(point.x) for point in points]
    ys = [float(point.y) for point in points]
    if not xs:
        return 0.0
    return max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))


def _handedness_entry(results: Any, index: int):
    handedness = getattr(results, 'multi_handedness', None) or []
    if index >= len(handedness):
        return None
    classification = getattr(handedness[index], 'classification', None) or []
    return classification[0] if classification else None


def hand_candidates(results: Any, image_is_mirrored: bool = False) -> List[HandCandidate]:
    """Converte a saída do MediaPipe em candidatas, na ordem original."""
    multi_hand_landmarks = getattr(results, 'multi_hand_landmarks', None) or []
    candidates: List[HandCandidate] = []
    for index, hand_landmarks in enumerate(multi_hand_landmarks):
        points = hand_landmarks.landmark
        entry = _handedness_entry(results, index)
        score = float(entry.score) if entry is not None else None
        raw_label = str(entry.label) if entry is not None else None
        normalized = _normalize_hand(raw_label)
        person_hand = normalized if image_is_mirrored else _swap_hand(normalized)
        wrist_point = points[WRIST_INDEX]
        candidates.append(
            HandCandidate(
                index=index,
                hand_landmarks=hand_landmarks,
                score=score,
                mediapipe_label=raw_label,
                person_hand=person_hand,
                bbox_area=_bbox_area(points),
                wrist=(float(wrist_point.x), float(wrist_point.y)),
            )
        )
    return candidates


def _eligible(candidates: List[HandCandidate], strategy: str, dominant_hand: Optional[str]) -> List[HandCandidate]:
    """Para ``dominant``, restringe às mãos do lado certo quando houver alguma."""
    if strategy != STRATEGY_DOMINANT or dominant_hand is None:
        return candidates
    matching = [candidate for candidate in candidates if candidate.person_hand == dominant_hand]
    return matching or candidates


def choose_candidate(
    candidates: List[HandCandidate],
    strategy: str = STRATEGY_LARGEST,
    dominant_hand: Optional[str] = None,
) -> Optional[HandCandidate]:
    """Aplica a estratégia sem histórico. Empates ficam com a ordem do MediaPipe."""
    if strategy not in STRATEGIES:
        raise ValueError(f'Estratégia de seleção de mão desconhecida: {strategy!r}; use uma de {STRATEGIES}')
    if not candidates:
        return None
    if strategy == STRATEGY_FIRST:
        return candidates[0]

    dominant = _normalize_hand(dominant_hand)
    pool = _eligible(candidates, strategy, dominant)
    if strategy == STRATEGY_MOST_CONFIDENT:
        # max() devolve o primeiro em caso de empate: mantém a ordem original.
        return max(pool, key=lambda candidate: candidate.score if candidate.score is not None else -1.0)
    # 'largest' e o desempate de 'dominant' (duas mãos do mesmo lado, ou nenhuma).
    return max(pool, key=lambda candidate: candidate.bbox_area)


def _distance(a: tuple, b: tuple) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class HandSelector:
    """Seleção com estado: estratégia na primeira detecção, rastreamento depois.

    Uso típico, uma instância por fluxo de vídeo::

        selector = HandSelector.from_config(HAND_SELECTION_CONFIG)
        candidate = selector.select(results)
        if candidate is not None:
            features = extract_landmarks_by_mode(candidate.landmarks, mode)
    """

    def __init__(
        self,
        strategy: str = DEFAULT_CONFIG['strategy'],
        dominant_hand: Optional[str] = DEFAULT_CONFIG['dominant_hand'],
        image_is_mirrored: bool = DEFAULT_CONFIG['image_is_mirrored'],
        tracking_max_jump: Optional[float] = DEFAULT_CONFIG['tracking_max_jump'],
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f'Estratégia de seleção de mão desconhecida: {strategy!r}; use uma de {STRATEGIES}')
        self.strategy = strategy
        self.dominant_hand = _normalize_hand(dominant_hand)
        self.image_is_mirrored = bool(image_is_mirrored)
        self.tracking_max_jump = tracking_max_jump
        self._last_wrist: Optional[tuple] = None

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None, **overrides) -> 'HandSelector':
        merged = dict(DEFAULT_CONFIG)
        merged.update(config or {})
        merged.update({key: value for key, value in overrides.items() if value is not None})
        return cls(
            strategy=merged['strategy'],
            dominant_hand=merged['dominant_hand'],
            image_is_mirrored=merged['image_is_mirrored'],
            tracking_max_jump=merged['tracking_max_jump'],
        )

    def reset(self) -> None:
        """Esquece a mão anterior (ex.: ao começar outra sequência ou vídeo)."""
        self._last_wrist = None

    def _tracked(self, pool: List[HandCandidate]) -> Optional[HandCandidate]:
        if self._last_wrist is None or not self.tracking_max_jump:
            return None
        nearest = min(pool, key=lambda candidate: _distance(candidate.wrist, self._last_wrist))
        if _distance(nearest.wrist, self._last_wrist) <= self.tracking_max_jump:
            return nearest
        return None

    def select(self, results: Any) -> Optional[HandCandidate]:
        candidates = hand_candidates(results, image_is_mirrored=self.image_is_mirrored)
        if not candidates:
            self.reset()
            return None

        if self.strategy == STRATEGY_FIRST:
            chosen = candidates[0]
        else:
            pool = _eligible(candidates, self.strategy, self.dominant_hand)
            chosen = self._tracked(pool) or choose_candidate(pool, self.strategy, self.dominant_hand)

        self._last_wrist = chosen.wrist
        return chosen


def build_hand_selector(config: Optional[Dict[str, Any]] = None, **overrides) -> HandSelector:
    """Atalho para ``HandSelector.from_config``."""
    return HandSelector.from_config(config, **overrides)
