"""
Divisão treino/validação/teste por pessoa
=========================================

Dividir amostras aleatoriamente faz a mesma pessoa aparecer nos três conjuntos
e infla a acurácia: o modelo aprende a pessoa, não o sinal. Aqui a unidade de
divisão é o ``subject_id``, garantindo que nenhuma pessoa cruze conjuntos.

A atribuição é determinística (mesma entrada, mesmo resultado) e gulosa: as
pessoas com mais amostras são alocadas primeiro ao conjunto com o maior déficit
em relação à sua cota, o que mantém as proporções próximas do alvo mesmo com
poucas pessoas e volumes desiguais.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

TRAIN = 'train'
VALIDATION = 'validation'
TEST = 'test'
SPLIT_NAMES = (TRAIN, VALIDATION, TEST)

DEFAULT_RATIOS = {TRAIN: 0.6, VALIDATION: 0.2, TEST: 0.2}


@dataclass
class SubjectSplit:
    """Resultado da divisão: pessoas e índices de amostra por conjunto."""

    subjects: Dict[str, List[str]] = field(default_factory=dict)
    indices: Dict[str, List[int]] = field(default_factory=dict)

    @property
    def train_indices(self) -> List[int]:
        return self.indices.get(TRAIN, [])

    @property
    def validation_indices(self) -> List[int]:
        return self.indices.get(VALIDATION, [])

    @property
    def test_indices(self) -> List[int]:
        return self.indices.get(TEST, [])

    def split_of_subject(self, subject_id: str) -> Optional[str]:
        for split_name, members in self.subjects.items():
            if subject_id in members:
                return split_name
        return None

    def summary(self) -> Dict[str, Dict[str, int]]:
        return {
            split_name: {
                'subjects': len(self.subjects.get(split_name, [])),
                'samples': len(self.indices.get(split_name, [])),
            }
            for split_name in SPLIT_NAMES
        }


def _normalize_ratios(ratios: Optional[Dict[str, float]]) -> Dict[str, float]:
    resolved = dict(DEFAULT_RATIOS if ratios is None else ratios)

    unknown = set(resolved).difference(SPLIT_NAMES)
    if unknown:
        raise ValueError(f'Conjuntos desconhecidos nas proporções: {sorted(unknown)}')

    for split_name in SPLIT_NAMES:
        resolved.setdefault(split_name, 0.0)
        if resolved[split_name] < 0:
            raise ValueError(f'Proporção negativa para {split_name}')

    total = sum(resolved.values())
    if total <= 0:
        raise ValueError('A soma das proporções deve ser maior que zero')

    return {name: value / total for name, value in resolved.items()}


def split_subjects_by_person(
    subject_ids: Sequence[str],
    ratios: Optional[Dict[str, float]] = None,
    allow_empty_splits: bool = False,
) -> SubjectSplit:
    """Divide amostras por pessoa.

    ``subject_ids[i]`` é a pessoa que gerou a amostra ``i``. Retorna um
    :class:`SubjectSplit` com as pessoas e os índices de cada conjunto.
    """
    resolved_ratios = _normalize_ratios(ratios)

    counts: Dict[str, int] = {}
    for subject_id in subject_ids:
        key = str(subject_id)
        counts[key] = counts.get(key, 0) + 1

    if not counts:
        raise ValueError('Nenhuma amostra fornecida para divisão')

    active_splits = [name for name in SPLIT_NAMES if resolved_ratios[name] > 0]
    if not allow_empty_splits and len(counts) < len(active_splits):
        raise ValueError(
            f'São necessárias ao menos {len(active_splits)} pessoas distintas para dividir '
            f'sem vazamento; encontradas {len(counts)}: {sorted(counts)}. '
            'Colete mais pessoas ou use allow_empty_splits=True.'
        )

    total_samples = sum(counts.values())
    targets = {name: resolved_ratios[name] * total_samples for name in SPLIT_NAMES}
    assigned = {name: 0 for name in SPLIT_NAMES}
    subjects: Dict[str, List[str]] = {name: [] for name in SPLIT_NAMES}

    # Pessoas com mais amostras primeiro; nome como desempate determinístico.
    ordered_subjects = sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    for position, (subject_id, sample_count) in enumerate(ordered_subjects):
        # Guloso pelo maior déficit deixaria conjuntos vazios quando há poucas
        # pessoas: quando só restam pessoas suficientes para cobrir os conjuntos
        # ainda vazios, elas vão obrigatoriamente para eles.
        remaining = len(ordered_subjects) - position
        empty_splits = [name for name in active_splits if not subjects[name]]
        candidates = empty_splits if 0 < len(empty_splits) >= remaining else active_splits

        # Maior déficit primeiro; a ordem de SPLIT_NAMES desempata.
        best_split = max(
            candidates,
            key=lambda name: (targets[name] - assigned[name], -SPLIT_NAMES.index(name)),
        )
        subjects[best_split].append(subject_id)
        assigned[best_split] += sample_count

    indices: Dict[str, List[int]] = {name: [] for name in SPLIT_NAMES}
    subject_to_split = {
        subject_id: split_name
        for split_name, members in subjects.items()
        for subject_id in members
    }
    for index, subject_id in enumerate(subject_ids):
        indices[subject_to_split[str(subject_id)]].append(index)

    for split_name in SPLIT_NAMES:
        subjects[split_name].sort()

    split = SubjectSplit(subjects=subjects, indices=indices)

    if not allow_empty_splits:
        empty = [name for name in active_splits if not subjects[name]]
        if empty:
            raise ValueError(f'Conjuntos vazios após a divisão: {empty}')

    return split


def split_metadata_by_person(
    metadata_list: Sequence[object],
    ratios: Optional[Dict[str, float]] = None,
    allow_empty_splits: bool = False,
) -> SubjectSplit:
    """Versão de :func:`split_subjects_by_person` para objetos com ``subject_id``."""
    subject_ids = [getattr(metadata, 'subject_id') for metadata in metadata_list]
    return split_subjects_by_person(subject_ids, ratios, allow_empty_splits)


def find_subject_leakage(split: SubjectSplit) -> List[Tuple[str, List[str]]]:
    """Retorna as pessoas que aparecem em mais de um conjunto."""
    membership: Dict[str, List[str]] = {}
    for split_name in SPLIT_NAMES:
        for subject_id in split.subjects.get(split_name, []):
            membership.setdefault(subject_id, []).append(split_name)

    return sorted(
        (subject_id, split_names)
        for subject_id, split_names in membership.items()
        if len(split_names) > 1
    )


def assert_no_subject_leakage(split: SubjectSplit) -> None:
    """Falha se alguma pessoa aparecer em mais de um conjunto."""
    leaks = find_subject_leakage(split)
    if leaks:
        details = ', '.join(f'{subject} em {names}' for subject, names in leaks)
        raise ValueError(f'Vazamento de pessoa entre conjuntos: {details}')
