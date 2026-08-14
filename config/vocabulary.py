"""
Vocabulário do LibrIA
=====================

Define de forma explícita o vocabulário reconhecido pelo sistema, separado em
três famílias (Fase 1 do plano arquitetural):

* ``alphabet``   - alfabeto manual (soletração).
* ``lexical``    - sinais de palavras completas.
* ``functional`` - gestos de controle (espaço, pausa, apagar, confirmar).

Cada entrada declara também a modalidade (``static`` ou ``temporal``), que
determina qual pipeline de coleta e de treino é responsável pela classe.

Este módulo é a fonte única de verdade do vocabulário: ``config.settings``
deriva suas listas de labels a partir daqui, e não o contrário. Por isso ele
não deve importar ``config.settings``.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Famílias de sinais
SIGN_TYPE_ALPHABET = 'alphabet'
SIGN_TYPE_LEXICAL = 'lexical'
SIGN_TYPE_FUNCTIONAL = 'functional'
SIGN_TYPES = (SIGN_TYPE_ALPHABET, SIGN_TYPE_LEXICAL, SIGN_TYPE_FUNCTIONAL)

# Modalidades de reconhecimento
MODALITY_STATIC = 'static'
MODALITY_TEMPORAL = 'temporal'
MODALITIES = (MODALITY_STATIC, MODALITY_TEMPORAL)

# Classe explícita de rejeição: usada tanto como rótulo de coleta negativa
# quanto como saída da inferência quando a confiança fica abaixo do limiar.
UNKNOWN_LABEL = 'DESCONHECIDO'


@dataclass(frozen=True)
class VocabularyEntry:
    """Uma classe do vocabulário.

    ``label`` é o identificador usado como nome de diretório no dataset e como
    rótulo do modelo, portanto precisa ser ASCII, maiúsculo e sem espaços.
    """

    label: str
    sign_type: str
    modality: str
    translation: str
    description: str = ''

    def __post_init__(self):
        if self.sign_type not in SIGN_TYPES:
            raise ValueError(f'sign_type inválido para {self.label}: {self.sign_type}')
        if self.modality not in MODALITIES:
            raise ValueError(f'modality inválida para {self.label}: {self.modality}')


def _alphabet_entries() -> List[VocabularyEntry]:
    """Alfabeto manual: A-Z, com J e Z como sinais temporais."""
    entries = []
    for code in range(ord('A'), ord('Z') + 1):
        letter = chr(code)
        modality = MODALITY_TEMPORAL if letter in {'J', 'Z'} else MODALITY_STATIC
        entries.append(
            VocabularyEntry(
                label=letter,
                sign_type=SIGN_TYPE_ALPHABET,
                modality=modality,
                translation=letter,
                description=f'Letra {letter} do alfabeto manual',
            )
        )
    return entries


# Conjunto lexical inicial: pequeno, útil e de alta frequência conversacional.
# Amplie esta lista conforme o dataset temporal crescer (Fase 3 do plano).
_LEXICAL_ENTRIES = [
    VocabularyEntry('OI', SIGN_TYPE_LEXICAL, MODALITY_TEMPORAL, 'oi'),
    VocabularyEntry('TUDO_BEM', SIGN_TYPE_LEXICAL, MODALITY_TEMPORAL, 'tudo bem'),
    VocabularyEntry('OBRIGADO', SIGN_TYPE_LEXICAL, MODALITY_TEMPORAL, 'obrigado'),
    VocabularyEntry('POR_FAVOR', SIGN_TYPE_LEXICAL, MODALITY_TEMPORAL, 'por favor'),
    VocabularyEntry('DESCULPA', SIGN_TYPE_LEXICAL, MODALITY_TEMPORAL, 'desculpa'),
    VocabularyEntry('SIM', SIGN_TYPE_LEXICAL, MODALITY_TEMPORAL, 'sim'),
    VocabularyEntry('NAO', SIGN_TYPE_LEXICAL, MODALITY_TEMPORAL, 'não'),
    VocabularyEntry('EU', SIGN_TYPE_LEXICAL, MODALITY_TEMPORAL, 'eu'),
    VocabularyEntry('VOCE', SIGN_TYPE_LEXICAL, MODALITY_TEMPORAL, 'você'),
    VocabularyEntry('NOME', SIGN_TYPE_LEXICAL, MODALITY_TEMPORAL, 'nome'),
    VocabularyEntry('AJUDA', SIGN_TYPE_LEXICAL, MODALITY_TEMPORAL, 'ajuda'),
    VocabularyEntry('ENTENDER', SIGN_TYPE_LEXICAL, MODALITY_TEMPORAL, 'entender'),
]

# Gestos funcionais: não viram palavra, viram comando na camada de composição.
_FUNCTIONAL_ENTRIES = [
    VocabularyEntry(
        'ESPACO', SIGN_TYPE_FUNCTIONAL, MODALITY_TEMPORAL, ' ',
        'Separa palavras durante a soletração',
    ),
    VocabularyEntry(
        'PAUSA', SIGN_TYPE_FUNCTIONAL, MODALITY_TEMPORAL, '',
        'Suspende o acúmulo de tokens sem encerrar a sessão',
    ),
    VocabularyEntry(
        'APAGAR', SIGN_TYPE_FUNCTIONAL, MODALITY_TEMPORAL, '',
        'Remove o último token confirmado',
    ),
    VocabularyEntry(
        'CONFIRMAR', SIGN_TYPE_FUNCTIONAL, MODALITY_TEMPORAL, '',
        'Fecha a sentença atual e envia para tradução',
    ),
]

# Classe de rejeição. Fica fora das três famílias porque não é um sinal: é o
# destino de tudo que o modelo não deve afirmar.
UNKNOWN_ENTRY = VocabularyEntry(
    UNKNOWN_LABEL,
    SIGN_TYPE_FUNCTIONAL,
    MODALITY_TEMPORAL,
    '',
    'Fora do vocabulário / não reconhecido',
)

VOCABULARY: Tuple[VocabularyEntry, ...] = tuple(
    _alphabet_entries() + _LEXICAL_ENTRIES + _FUNCTIONAL_ENTRIES
)

_VOCABULARY_BY_LABEL: Dict[str, VocabularyEntry] = {
    entry.label: entry for entry in VOCABULARY
}


def get_entry(label: str) -> Optional[VocabularyEntry]:
    """Retorna a entrada do vocabulário, ou ``None`` se for fora do vocabulário."""
    if label is None:
        return None
    normalized = str(label).strip().upper()
    if normalized == UNKNOWN_LABEL:
        return UNKNOWN_ENTRY
    return _VOCABULARY_BY_LABEL.get(normalized)


def is_known_label(label: str) -> bool:
    """Informa se o rótulo pertence ao vocabulário (a classe de rejeição não pertence)."""
    normalized = str(label).strip().upper() if label is not None else ''
    return normalized in _VOCABULARY_BY_LABEL


def get_labels(sign_type: Optional[str] = None, modality: Optional[str] = None) -> List[str]:
    """Lista labels do vocabulário filtrando por família e/ou modalidade."""
    if sign_type is not None and sign_type not in SIGN_TYPES:
        raise ValueError(f'sign_type inválido: {sign_type}')
    if modality is not None and modality not in MODALITIES:
        raise ValueError(f'modality inválida: {modality}')

    return [
        entry.label
        for entry in VOCABULARY
        if (sign_type is None or entry.sign_type == sign_type)
        and (modality is None or entry.modality == modality)
    ]


def get_sign_type(label: str) -> Optional[str]:
    """Retorna a família do rótulo, ou ``None`` se for fora do vocabulário."""
    entry = get_entry(label)
    return entry.sign_type if entry is not None else None


def get_modality(label: str) -> Optional[str]:
    """Retorna a modalidade do rótulo, ou ``None`` se for fora do vocabulário."""
    entry = get_entry(label)
    return entry.modality if entry is not None else None


def get_translation(label: str) -> str:
    """Retorna a tradução em português do rótulo (vazio para gestos funcionais)."""
    entry = get_entry(label)
    return entry.translation if entry is not None else ''


def validate_vocabulary() -> List[str]:
    """Valida a consistência do vocabulário e retorna a lista de erros."""
    errors = []

    seen = set()
    for entry in VOCABULARY:
        if entry.label in seen:
            errors.append(f'Label duplicada no vocabulário: {entry.label}')
        seen.add(entry.label)

        if entry.label != entry.label.upper():
            errors.append(f'Label deve ser maiúscula: {entry.label}')
        if not entry.label.isascii():
            errors.append(f'Label deve ser ASCII (nome de diretório): {entry.label}')
        if ' ' in entry.label:
            errors.append(f'Label não pode conter espaço: {entry.label}')

    if UNKNOWN_LABEL in seen:
        errors.append(
            f'{UNKNOWN_LABEL} é a classe de rejeição e não deve estar no vocabulário'
        )

    alphabet = get_labels(sign_type=SIGN_TYPE_ALPHABET)
    if len(alphabet) != 26:
        errors.append('O alfabeto manual deve conter as 26 letras')

    return errors
