"""
Fixtures de paridade: pipeline Python → núcleo nativo (C++) e, depois, Android
==============================================================================

O pipeline Python é a referência de comportamento. Este script roda o código
Python **real** (não uma reimplementação) sobre entradas determinísticas e grava
entradas e saídas em JSON. O núcleo C++ (``native/libria_core``) lê esse arquivo
nos testes e precisa reproduzir os mesmos resultados.

Nos cenários do pipeline temporal, cada chamada aos preditores é gravada com a
entrada exata recebida e a resposta devolvida. Assim o teste nativo confere que
consulta os modelos nos mesmos quadros e com as mesmas janelas, e não apenas que
o resultado final coincide.

Uso::

    python -m scripts.export_parity_fixtures            # regrava o arquivo
    python -m scripts.export_parity_fixtures --check    # falha se estiver desatualizado

Os valores de configuração vêm de ``config/settings.py``: se um limiar mudar lá,
as fixtures precisam ser regeneradas (o ``--check`` acusa).
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    HYBRID_INFERENCE_CONFIG,
    TEMPORAL_PIPELINE_CONFIG,
)
from src.inference.libras_embedded_runtime import (  # noqa: E402
    EmbeddedPrediction,
    QuantizedTFLiteModel,
    choose_embedded_prediction,
)
from src.inference.motion_detector import MotionDetector  # noqa: E402
from src.inference.temporal_buffer import resample_sequence  # noqa: E402
from src.inference.temporal_pipeline import TemporalPipeline  # noqa: E402
from utils.helpers import landmarks_to_relative  # noqa: E402

FORMAT_VERSION = 1
DEFAULT_OUTPUT = PROJECT_ROOT / 'native' / 'fixtures' / 'parity_fixtures.json'

FPS = 30.0
PIPELINE_DIM = 6  # 2 pontos x 3 coordenadas: o pipeline não depende da dimensão
SEQUENCE_LENGTH = 30
# Distância relativa mínima entre uma energia e um limiar. Abaixo disso, a
# diferença de arredondamento entre float32 (NumPy) e double (C++) poderia virar
# a decisão, e a fixture seria frágil sem ninguém perceber.
THRESHOLD_MARGIN = 0.02
# Grade dos valores do fluxo sintético: múltiplos de 2^-16 são exatos em float32
# e têm representação decimal curta, o que reduz o tamanho do JSON.
GRID = 2.0 ** -16


def _f32(values) -> List[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float32).reshape(-1)]


def _f32_rows(values) -> List[List[float]]:
    array = np.asarray(values, dtype=np.float32)
    return [[float(v) for v in row] for row in array]


# ---------------------------------------------------------------------------
# Features (utils.helpers.landmarks_to_relative)
# ---------------------------------------------------------------------------

def _associativity_sensitive_hand(rng: np.random.RandomState) -> np.ndarray:
    """Mão cujo ponto mais distante tem norma sensível à ordem da soma.

    np.linalg.norm calcula sqrt((x*x + y*y) + z*z) em float32. Este caso garante
    que sqrt(x*x + (y*y + z*z)) dê outro valor, para que o teste nativo acuse
    uma implementação que some na ordem errada.
    """
    f32 = np.float32
    for _ in range(100000):
        x, y, z = (f32(v) for v in rng.uniform(0.2, 0.4, size=3))
        left = np.sqrt(f32(f32(f32(x * x) + f32(y * y)) + f32(z * z)))
        right = np.sqrt(f32(f32(x * x) + f32(f32(y * y) + f32(z * z))))
        if left != right:
            hand = rng.uniform(-0.05, 0.05, size=(21, 3))  # demais pontos bem mais perto
            hand[0] = 0.0                                   # pulso na origem: delta == ponto
            hand[7] = (x, y, z)
            return hand
    raise RuntimeError('não encontrei ponto sensível à associatividade')


def _landmark_cases() -> List[Dict]:
    rng = np.random.RandomState(1234)
    hands = []
    for _ in range(6):
        wrist = rng.uniform(0.2, 0.8, size=3) * np.array([1.0, 1.0, 0.0])
        points = wrist + rng.normal(0.0, 0.08, size=(21, 3)) * np.array([1.0, 1.0, 0.5])
        points[0] = wrist
        hands.append(points)
    hands.append(_associativity_sensitive_hand(rng))         # a ordem da soma na norma importa
    hands.append(np.full((21, 3), 0.5))                      # degenerada: max_distance == 0
    z_only = np.zeros((21, 3))
    z_only[:, 2] = np.linspace(0.0, 0.2, 21)
    hands.append(z_only)                                     # só profundidade varia

    cases = []
    for hand in hands:
        hand32 = np.asarray(hand, dtype=np.float32)
        # O MediaPipe entrega floats de 32 bits; o objeto imita .x/.y/.z.
        landmarks = [SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in hand32]
        cases.append({
            'landmarks': _f32_rows(hand32),
            'expected': _f32(landmarks_to_relative(landmarks)),
        })
    return cases


# ---------------------------------------------------------------------------
# Reamostragem (temporal_buffer.resample_sequence)
# ---------------------------------------------------------------------------

def _resample_cases() -> List[Dict]:
    pairs = [(n, 30) for n in (1, 2, 3, 5, 8, 9, 10, 15, 29, 30, 31, 45, 59, 60, 61, 90, 119, 120, 121)]
    pairs += [(10, 1), (7, 7), (13, 7), (5, 64), (100, 64)]
    # Empates exatos com piso par (2.5 -> 2, 7.5 -> 8): distinguem np.rint de round.
    pairs += [(6, 3), (11, 5)]
    cases = []
    for frames, target in pairs:
        source = np.arange(frames, dtype=np.float32).reshape(-1, 1)
        indices = resample_sequence(source, target)[:, 0].astype(int).tolist()
        cases.append({'frames': frames, 'target': target, 'indices': indices})
    return cases


# ---------------------------------------------------------------------------
# Detector de movimento (motion_detector.MotionDetector)
# ---------------------------------------------------------------------------

def _motion_cases() -> List[Dict]:
    rng = np.random.RandomState(99)
    cases = []

    def run(smoothing: float, frames: Sequence[Optional[np.ndarray]]) -> Dict:
        detector = MotionDetector(smoothing=smoothing)
        energies, readings = [], []
        for frame in frames:
            energies.append(float(detector.update(frame)))
            readings.append(bool(detector.has_reading))
        return {
            'smoothing': smoothing,
            'frames': [None if f is None else _f32(f) for f in frames],
            'energy': energies,
            'has_reading': readings,
        }

    base63 = rng.uniform(0.0, 1.0, size=63).astype(np.float32)
    walk63 = [base63 + np.float32(0.01) * k * rng.normal(size=63).astype(np.float32) for k in range(8)]
    cases.append(run(0.6, walk63[:4] + [None] + walk63[4:]))
    cases.append(run(0.0, walk63))
    cases.append(run(0.9, walk63))
    # Mudança de dimensão no meio (energia bruta 0) e dimensão não múltipla de 3.
    six = [rng.uniform(size=6).astype(np.float32) for _ in range(3)]
    nine = rng.uniform(size=9).astype(np.float32)
    cases.append(run(0.5, six + [nine, nine + np.float32(0.05)]))
    four = [rng.uniform(size=4).astype(np.float32) for _ in range(4)]
    cases.append(run(0.3, four))
    return cases


# ---------------------------------------------------------------------------
# Regra híbrida (libras_embedded_runtime.choose_embedded_prediction)
# ---------------------------------------------------------------------------

def _hybrid_cases() -> List[Dict]:
    static_thr = float(HYBRID_INFERENCE_CONFIG['static_confidence_threshold'])
    temporal_thr = float(HYBRID_INFERENCE_CONFIG['temporal_confidence_threshold'])
    priority = [str(label) for label in HYBRID_INFERENCE_CONFIG['temporal_priority_classes']]

    def pred(token, confidence, source):
        return None if token is None else EmbeddedPrediction(token, confidence, source)

    combos = [
        (('A', 0.80), ('J', 0.90)),
        (('A', 0.80), ('J', 0.80)),
        (('A', 0.80), ('j', 0.90)),
        (('A', 0.70), ('X', 0.95)),
        (('A', 0.75), None),
        (None, None),
        (None, ('Z', 0.85)),
        (('B', 0.74), ('Z', 0.84)),
    ]
    cases = []
    for static, temporal in combos:
        s = pred(*(static or (None, None)), 'static')
        t = pred(*(temporal or (None, None)), 'temporal')
        chosen = choose_embedded_prediction(s, t, static_thr, temporal_thr, priority)
        cases.append({
            'static': None if s is None else {'token': s.token, 'confidence': s.confidence},
            'temporal': None if t is None else {'token': t.token, 'confidence': t.confidence},
            'static_threshold': static_thr,
            'temporal_threshold': temporal_thr,
            'temporal_priority_classes': priority,
            'expected': None if chosen is None else {
                'token': chosen.token, 'confidence': chosen.confidence, 'source': chosen.source,
            },
        })
    return cases


# ---------------------------------------------------------------------------
# Quantização int8 (libras_embedded_runtime.QuantizedTFLiteModel.predict)
# ---------------------------------------------------------------------------

class _FakeInterpreter:
    """Intérprete falso: captura a entrada quantizada e devolve uma saída pronta."""

    def __init__(self, raw_output: np.ndarray):
        self.raw_output = raw_output
        self.captured: Optional[np.ndarray] = None

    def set_tensor(self, _index, value):
        self.captured = np.array(value, copy=True)

    def invoke(self):
        pass

    def get_tensor(self, _index):
        return self.raw_output


def _quantization_cases() -> List[Dict]:
    rng = np.random.RandomState(7)
    specs = [
        # (entrada, escala/zero da entrada, saída crua int8, escala/zero da saída)
        (rng.uniform(-1.0, 1.0, size=(21, 3)), (1.0 / 128.0, 0), [-128, 50, 127], (1.0 / 256.0, -128)),
        # Empates .5 (np.round arredonda para o par) e clip no intervalo do int8.
        (np.array([0.25, 0.75, -0.25, -0.75, 1.25, 2.5, 1000.0, -1000.0]), (0.5, 0), [10, 20], (0.1, 0)),
        # Escala não representável exatamente em float32 e zero point não nulo.
        (rng.uniform(-0.5, 0.5, size=(30, 6)), (0.02, -3), [-100, 0, 100, 27], (0.02, -3)),
        # Escala 0: sem quantização na entrada nem desquantização na saída.
        (np.array([1.7, -1.7, 3.2, 0.4]), (0.0, 0), [3, 1], (0.0, 0)),
        # Saída toda no zero point: soma 0, devolvida sem normalizar.
        (np.array([0.1, 0.2]), (0.5, 0), [-128, -128, -128], (1.0 / 256.0, -128)),
    ]
    cases = []
    for tensor, (in_scale, in_zero), raw, (out_scale, out_zero) in specs:
        tensor32 = np.asarray(tensor, dtype=np.float32)
        raw_output = np.asarray([raw], dtype=np.int8)
        fake = _FakeInterpreter(raw_output)
        model = QuantizedTFLiteModel.__new__(QuantizedTFLiteModel)
        model.interpreter = fake
        model.input_details = {'index': 0, 'quantization': (in_scale, in_zero), 'dtype': np.int8}
        model.output_details = {'index': 1, 'quantization': (out_scale, out_zero), 'dtype': np.int8}
        probabilities = model.predict(tensor32)
        cases.append({
            'input': _f32(tensor32),
            'input_scale': float(in_scale),
            'input_zero_point': int(in_zero),
            'quantized': [int(v) for v in fake.captured.reshape(-1)],
            'raw_output': [int(v) for v in raw_output.reshape(-1)],
            'output_scale': float(out_scale),
            'output_zero_point': int(out_zero),
            'probabilities': _f32(probabilities),
        })
    return cases


# ---------------------------------------------------------------------------
# Pipeline temporal (temporal_pipeline.TemporalPipeline)
# ---------------------------------------------------------------------------

class _Stream:
    """Gera um fluxo determinístico de quadros: mão parada, movimento ou ausente."""

    def __init__(self, dim: int = PIPELINE_DIM):
        self.position = self._snap(np.linspace(0.3, 0.7, dim))
        self.frames: List[Optional[np.ndarray]] = []

    @staticmethod
    def _snap(values: np.ndarray) -> np.ndarray:
        return np.round(np.asarray(values, dtype=np.float64) / GRID) * GRID

    def _push(self, values: np.ndarray) -> None:
        # Assinatura por quadro na última coordenada (k * GRID, estritamente
        # crescente): torna cada quadro único, para as chamadas aos preditores
        # serem gravadas como índices de quadro sem ambiguidade. O efeito na
        # energia é desprezível (~1e-5) e a margem dos limiares é conferida.
        frame = self._snap(values)
        frame[-1] += len(self.frames) * GRID
        self.frames.append(frame.astype(np.float32))

    def hold(self, count: int) -> '_Stream':
        for _ in range(count):
            k = len(self.frames)
            # Tremor mínimo e determinístico: mão parada nunca é estática pixel a pixel.
            jitter = 1e-4 * np.sin(1.7 * k + np.arange(self.position.size))
            self._push(self.position + jitter)
        return self

    def move(self, count: int, speed: float = 9.0 / 256.0) -> '_Stream':
        direction = np.tile([1.0, 0.0, 0.0], self.position.size // 3)
        for step in range(count):
            # Zigue-zague: inverte o sentido a cada 10 quadros, sem parar.
            sign = 1.0 if (step // 10) % 2 == 0 else -1.0
            self.position = self._snap(self.position + sign * speed * direction)
            self._push(self.position)
        return self

    def absent(self, count: int) -> '_Stream':
        self.frames.extend([None] * count)
        return self


class _Recorder:
    """Envolve um preditor roteirizado e grava cada chamada (entrada e saída).

    A entrada é gravada como índices dos quadros do cenário: o modelo temporal
    recebe uma lista de quadros (janela ou segmento reamostrado) e o estático,
    um quadro só.
    """

    def __init__(self, respond: Callable[[int], object], kind: str, lookup: Dict[tuple, int]):
        self.respond = respond
        self.kind = kind
        self.lookup = lookup
        self.calls: List[Dict] = []

    def _index_of(self, row: np.ndarray) -> int:
        key = tuple(float(v) for v in row)
        if key not in self.lookup:
            raise RuntimeError('preditor recebeu um quadro que não está no cenário')
        return self.lookup[key]

    def __call__(self, value):
        output = self.respond(len(self.calls))
        array = np.asarray(value, dtype=np.float32)
        if self.kind == 'temporal':
            recorded_input = [self._index_of(row) for row in array]
        else:
            recorded_input = self._index_of(array.reshape(-1))
        if self.kind == 'temporal':
            recorded_output = [float(p) for p in output]
        else:
            recorded_output = {'label': output[0], 'confidence': float(output[1])}
        self.calls.append({'input_frames': recorded_input, 'output': recorded_output})
        return output


def _token_dict(token) -> Optional[Dict]:
    if token is None:
        return None
    return {
        'label': token.label,
        'confidence': token.confidence,
        'start_time': token.start_time,
        'end_time': token.end_time,
        'source': token.source,
        'state': token.state,
        'frame_count': token.frame_count,
    }


def _check_margin(name: str, energies: Sequence[float], config: Dict) -> None:
    for threshold_key in ('motion_start_threshold', 'motion_end_threshold'):
        threshold = float(config[threshold_key])
        for index, energy in enumerate(energies):
            if abs(energy - threshold) / threshold < THRESHOLD_MARGIN:
                raise RuntimeError(
                    f'cenário {name!r}, quadro {index}: energia {energy:.6g} a menos de '
                    f'{THRESHOLD_MARGIN:.0%} de {threshold_key}={threshold}; ajuste o cenário'
                )


def _run_scenario(
    name: str,
    stream: _Stream,
    temporal_respond: Callable[[int], Sequence[float]],
    static_respond: Optional[Callable[[int], Tuple[str, float]]],
    label_map: Dict[int, str],
    overrides: Optional[Dict] = None,
) -> Dict:
    config = dict(TEMPORAL_PIPELINE_CONFIG)
    config.update(overrides or {})

    lookup = {
        tuple(float(v) for v in frame): index
        for index, frame in enumerate(stream.frames) if frame is not None
    }
    present = sum(frame is not None for frame in stream.frames)
    if len(lookup) != present:
        raise RuntimeError(f'cenário {name!r}: quadros repetidos; a gravação por índice seria ambígua')

    temporal = _Recorder(temporal_respond, 'temporal', lookup)
    static = _Recorder(static_respond, 'static', lookup) if static_respond is not None else None
    pipeline = TemporalPipeline(
        temporal_predictor=temporal,
        label_map=label_map,
        sequence_length=SEQUENCE_LENGTH,
        static_predictor=static,
        config=config,
    )

    timestamps, expected = [], []
    for index, features in enumerate(stream.frames):
        timestamp = index / FPS
        token = pipeline.process_frame(features, timestamp, hand_present=features is not None)
        timestamps.append(timestamp)
        expected.append({
            'energy': float(pipeline.last_energy),
            'signing': bool(pipeline.is_signing),
            'token': _token_dict(token),
        })

    _check_margin(name, [e['energy'] for e in expected], config)
    return {
        'name': name,
        'sequence_length': SEQUENCE_LENGTH,
        'feature_dim': PIPELINE_DIM,
        'config': {key: config[key] for key in sorted(config)},
        'label_map': {str(k): v for k, v in sorted(label_map.items())},
        'has_static_predictor': static is not None,
        'frames': [
            {'t': t, 'f': None if f is None else _f32(f)}
            for t, f in zip(timestamps, stream.frames)
        ],
        'temporal_calls': temporal.calls,
        'static_calls': [] if static is None else static.calls,
        'expected': expected,
    }


def _pipeline_scenarios() -> List[Dict]:
    jz = {0: 'J', 1: 'Z'}

    def constant(probabilities):
        return lambda _call: list(probabilities)

    def ramp_then(final_after: int, early, late):
        return lambda call: list(early if call < final_after else late)

    def cycle(responses):
        return lambda call: responses[call % len(responses)]

    scenarios = [
        # Só mão parada: fallback estático com strip/upper, limiar e deduplicação.
        _run_scenario(
            'static_hold', _Stream().hold(140), constant([0.5, 0.5]),
            cycle([('a ', 0.90), ('A', 0.95), ('B', 0.50), ('B', 0.80), ('B', 0.85), ('A', 0.90), ('A', 0.92)]),
            jz,
        ),
        # Sinal completo: parciais a cada 5 quadros, final suavizado, cooldown do estático.
        _run_scenario(
            'sign_then_rest', _Stream().hold(40).move(40).hold(60),
            ramp_then(6, [0.7, 0.3], [0.95, 0.05]),
            constant(('A', 0.90)), jz,
        ),
        # Estático consultado a cada 5 quadros: logo após o sinal, a mão parada é
        # bloqueada pelo cooldown (não pela energia) e só depois volta a valer.
        _run_scenario(
            'static_cooldown', _Stream().hold(40).move(40).hold(60),
            constant([0.95, 0.05]), constant(('A', 0.90)), jz,
            overrides={'static_interval_frames': 5},
        ),
        # Confiança abaixo do limiar: token recusado (DESCONHECIDO); sem preditor estático.
        _run_scenario(
            'low_confidence_sign', _Stream().hold(35).move(30).hold(40),
            constant([0.6, 0.4]), None, jz,
        ),
        # Mão some no meio do sinal: fecha por hand_lost.
        _run_scenario(
            'hand_lost', _Stream().hold(35).move(30).absent(10).hold(30),
            constant([0.1, 0.9]), constant(('B', 0.90)), jz,
        ),
        # Movimento contínuo além de max_duration: fecha e reabre em seguida.
        _run_scenario(
            'max_duration', _Stream().hold(32).move(70).hold(40),
            ramp_then(10, [0.9, 0.1], [0.1, 0.9]), None, jz,
            overrides={'max_duration_seconds': 1.5},
        ),
        # Dois J seguidos em menos de 1 s: o segundo é suprimido; o terceiro, não.
        _run_scenario(
            'duplicate_temporal',
            _Stream().hold(32).move(20).hold(8).move(12).hold(40).move(12).hold(40),
            constant([0.92, 0.08]), None, jz,
        ),
        # Gesto curto: o recorte dos quadros parados não se aplica (ficaria < min_frames).
        _run_scenario(
            'blip_trim_skipped', _Stream().hold(32).move(2).hold(30),
            constant([0.9, 0.1]), None, jz,
        ),
        # Mesmo gesto curto com min_segment_frames maior: o segmento é descartado.
        _run_scenario(
            'blip_dropped', _Stream().hold(32).move(2).hold(30),
            constant([0.9, 0.1]), None, jz, overrides={'min_segment_frames': 12},
        ),
        # Ausências curtas com a mão parada: has_reading reinicia e o estático espera.
        _run_scenario(
            'idle_absent_gaps', _Stream().hold(25).absent(3).hold(30).absent(1).hold(25),
            constant([0.5, 0.5]), constant(('C', 0.90)), jz,
        ),
        # Parciais desligadas: a decisão final usa só a predição do segmento.
        _run_scenario(
            'partials_disabled', _Stream().hold(40).move(40).hold(60),
            ramp_then(1, [0.95, 0.05], [0.3, 0.7]), None, jz,
            overrides={'emit_partial_tokens': False},
        ),
        # label_map minúsculo: o supressor compara em maiúsculas, então o "A"
        # estático logo após o "a" temporal é suprimido (e renovado).
        _run_scenario(
            'lowercase_label_map', _Stream().hold(40).move(40).hold(60),
            constant([0.95, 0.05]), constant(('A', 0.90)), {0: 'a', 1: 'z'},
            overrides={'static_interval_frames': 5},
        ),
        # Espaços no label_map: o token sai sem espaços, mas o supressor recebe o
        # rótulo sem strip (" A"), então o "A" estático não é suprimido.
        _run_scenario(
            'label_with_spaces', _Stream().hold(40).move(40).hold(60),
            constant([0.95, 0.05]), constant(('A', 0.90)), {0: ' a', 1: 'z'},
            overrides={'static_interval_frames': 5},
        ),
        # Índice fora do label_map: o rótulo vira o próprio número.
        _run_scenario(
            'label_map_missing_index', _Stream().hold(35).move(30).hold(40),
            constant([0.05, 0.05, 0.9]), None, {0: 'J'},
        ),
    ]
    return scenarios


# ---------------------------------------------------------------------------

def build_fixtures() -> Dict:
    return {
        'format_version': FORMAT_VERSION,
        'generator': 'scripts/export_parity_fixtures.py',
        'features': _landmark_cases(),
        'resample': _resample_cases(),
        'motion': _motion_cases(),
        'hybrid_decision': _hybrid_cases(),
        'quantization': _quantization_cases(),
        'pipeline': _pipeline_scenarios(),
    }


def render(fixtures: Dict) -> str:
    return json.dumps(fixtures, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n'


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true',
                        help='não grava; falha se o arquivo existente estiver desatualizado')
    args = parser.parse_args(argv)

    content = render(build_fixtures())
    if args.check:
        current = args.output.read_text(encoding='utf-8') if args.output.exists() else ''
        if current != content:
            print(f'❌ {args.output} desatualizado. Rode: python -m scripts.export_parity_fixtures')
            return 1
        print(f'✅ {args.output} em dia com o pipeline Python')
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding='utf-8')
    print(f'✅ Fixtures gravadas em {args.output} ({len(content) / 1024:.0f} KiB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
