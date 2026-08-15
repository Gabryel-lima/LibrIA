"""
Fontes externas de dados de Libras
==================================

Coletar tudo na webcam não escala: cada classe nova custa uma sessão manual e
o dataset fica preso a uma pessoa, uma câmera e um ambiente. Este módulo é o
catálogo das bases públicas que podem entrar no dataset **sem** nova gravação.

O que entra aqui é só a descrição da fonte — o download e a conversão ficam em
``scripts/fetch_sources.py`` e ``src/dataset/video_ingest.py``. Toda fonte vira
o mesmo contrato do resto do projeto: ``.npy`` de landmarks + ``.json`` de
metadados, com ``source_dataset`` preenchido para dar rastreabilidade e permitir
dividir treino/teste por origem além de por pessoa.

Licença importa: ``license`` e ``requires_agreement`` dizem se a base pode ser
redistribuída ou apenas usada localmente. Nada aqui é baixado automaticamente
quando ``requires_agreement`` é ``True``.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config.vocabulary import MODALITY_STATIC, MODALITY_TEMPORAL

# Como obter os arquivos da fonte.
ACCESS_DIRECT = 'direct'        # download automatizável (URL pública estável)
ACCESS_ACCOUNT = 'account'      # exige conta na plataforma (Kaggle, IEEE DataPort)
ACCESS_REQUEST = 'request'      # exige pedido/aceite de termos com os autores

# Formato do que se baixa.
CONTENT_VIDEO = 'video'
CONTENT_IMAGE = 'image'
CONTENT_LANDMARK = 'landmark'


@dataclass(frozen=True)
class DataSource:
    """Uma base externa que pode alimentar o dataset do LibrIA."""

    key: str
    name: str
    language: str
    url: str
    content: str
    modality: str
    access: str
    license: str
    size: str = 'desconhecido'
    signers: Optional[int] = None
    classes: Optional[int] = None
    notes: str = ''
    # Comandos/passos para obter os arquivos, quando não há download direto.
    instructions: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def automatable(self) -> bool:
        """A fonte pode ser baixada por `make fetch` sem intervenção humana."""
        return self.access == ACCESS_DIRECT

    @property
    def requires_agreement(self) -> bool:
        return self.access == ACCESS_REQUEST


# ---------------------------------------------------------------------------
# Catálogo
# ---------------------------------------------------------------------------
# Ordenado por utilidade para o vocabulário atual: primeiro Libras temporal
# (palavras), depois alfabeto estático, depois bases de outras línguas de
# sinais — úteis só para pré-treino, nunca para avaliar Libras.

DATA_SOURCES: Tuple[DataSource, ...] = (
    DataSource(
        key='minds-libras',
        name='MINDS-Libras',
        language='libras',
        url='https://zenodo.org/record/2667329',
        content=CONTENT_VIDEO,
        modality=MODALITY_TEMPORAL,
        access=ACCESS_DIRECT,
        license='Creative Commons (ver registro no Zenodo)',
        size='~64.8 GB (RGB + profundidade)',
        signers=12,
        classes=20,
        notes=(
            'Base da UFMG: 20 sinais, 5 repetições, 12 sinalizantes, fundo chroma key, '
            '1920x1080 a 30 fps. É a melhor fonte para variação entre pessoas — '
            'exatamente o que a coleta em uma única webcam não dá. Baixe só os vídeos RGB.'
        ),
        instructions=(
            'O registro tem um ZIP por sinalizante (Sinalizador01..12, 2.5 a 8.4 GB cada).',
            'Comece por um: python -m scripts.fetch_sources minds-libras --filter Sinalizador01',
            'Descompacte em data/archives/minds-libras/ e confira como o sinal aparece '
            'no caminho (pasta ou nome de arquivo).',
            'O rótulo não está na pasta de topo — use --label-from filename (ou --label-regex) '
            'e --subject-pattern "Sinalizador(?P<subject>\\d+)" para preservar a pessoa.',
        ),
    ),
    DataSource(
        key='v-librasil',
        name='V-LIBRASIL',
        language='libras',
        url='https://libras.cin.ufpe.br/',
        content=CONTENT_VIDEO,
        modality=MODALITY_TEMPORAL,
        access=ACCESS_ACCOUNT,
        license='uso acadêmico (ver termos na plataforma)',
        size='~10.5 GB (3 partes)',
        signers=3,
        classes=1364,
        notes=(
            'UFPE: 1364 termos por 3 intérpretes (4089 sinais), chroma key. '
            'Maior vocabulário lexical disponível — cobre com folga as 12 palavras '
            'do vocabulário atual e permite ampliá-lo sem gravar nada.'
        ),
        instructions=(
            'Acesse https://libras.cin.ufpe.br/ e baixe as partes por articulador.',
            'Extraia em data/archives/v-librasil/ mantendo uma pasta por termo.',
            'make ingest SOURCE_DIR=data/archives/v-librasil MODALITY=temporal '
            'SOURCE_NAME=v-librasil LABEL_MAP=data/label_maps/v-librasil.json',
        ),
    ),
    DataSource(
        key='ufop-libras',
        name='LIBRAS-UFOP (pares mínimos)',
        language='libras',
        url='https://www.repositorio.ufop.br/handle/123456789/14751',
        content=CONTENT_VIDEO,
        modality=MODALITY_TEMPORAL,
        access=ACCESS_REQUEST,
        license='mediante solicitação aos autores',
        classes=56,
        notes=(
            'RGB-D + esqueleto (Kinect V1), 56 sinais escolhidos como pares mínimos — '
            'sinais que diferem em um único parâmetro. Excelente para medir os erros '
            'que mais importam, e não a acurácia média.'
        ),
        instructions=(
            'Solicite acesso pelo artigo/repositório da UFOP.',
            'Só os vídeos RGB são necessários para o pipeline atual.',
        ),
    ),
    DataSource(
        key='ines-dicionario',
        name='Dicionário da Língua Brasileira de Sinais (INES)',
        language='libras',
        url='https://www.ines.gov.br/dicionario-de-libras/',
        content=CONTENT_VIDEO,
        modality=MODALITY_TEMPORAL,
        access=ACCESS_REQUEST,
        license='direitos reservados ao INES — verificar uso antes de treinar',
        notes=(
            'Referência oficial de forma dos sinais. Útil para validar a articulação '
            'de cada classe do vocabulário antes de gravar; não trate como base de '
            'treino sem checar os termos de uso.'
        ),
        instructions=(
            'Consulte sinal a sinal no site para conferir a forma correta.',
        ),
    ),
    DataSource(
        key='libras-alphabet-roboflow',
        name='Alfabeto em Libras (Roboflow Universe)',
        language='libras',
        url='https://universe.roboflow.com/search?q=alfabeto+libras',
        content=CONTENT_IMAGE,
        modality=MODALITY_STATIC,
        access=ACCESS_ACCOUNT,
        license='varia por dataset (majoritariamente CC BY 4.0)',
        classes=26,
        notes=(
            'Vários conjuntos de imagens do alfabeto (1.5k a 4.4k imagens cada), com '
            'mãos, iluminações e fundos diferentes. Caminho mais barato para tirar o '
            'alfabeto estático da dependência de uma única webcam.'
        ),
        instructions=(
            'Baixe no formato "folder structure" (uma pasta por letra).',
            'Extraia em data/archives/<nome>/ e rode: '
            'make ingest SOURCE_DIR=data/archives/<nome> MODALITY=static SOURCE_NAME=<nome>',
        ),
    ),
    DataSource(
        key='bsl-alphabet-dataset',
        name='Brazilian Sign Language Alphabet Dataset',
        language='libras',
        url='https://biankatpas.github.io/Brazilian-Sign-Language-Alphabet-Dataset/',
        content=CONTENT_IMAGE,
        modality=MODALITY_STATIC,
        access=ACCESS_ACCOUNT,
        license='ver repositório',
        classes=26,
        size='4411 imagens',
        notes='Compilação de imagens do alfabeto de Libras organizada por letra.',
        instructions=(
            'Clone o repositório indicado na página e extraia em data/archives/.',
        ),
    ),
    DataSource(
        key='wlasl',
        name='WLASL (American Sign Language)',
        language='asl',
        url='https://dxli94.github.io/WLASL/',
        content=CONTENT_VIDEO,
        modality=MODALITY_TEMPORAL,
        access=ACCESS_ACCOUNT,
        license='Computational Use of Data Agreement (C-UDA)',
        classes=2000,
        notes=(
            'ASL, não Libras: serve para pré-treinar o extrator temporal e depois '
            'fazer fine-tuning em Libras. Nunca use para reportar acurácia em Libras — '
            'os sinais são de outra língua.'
        ),
        instructions=(
            'Baixe os metadados no site e os vídeos com o script oficial do projeto.',
        ),
    ),
)

SOURCES_BY_KEY: Dict[str, DataSource] = {source.key: source for source in DATA_SOURCES}


def get_source(key: str) -> Optional[DataSource]:
    return SOURCES_BY_KEY.get(str(key).strip().lower())


def list_sources(
    language: Optional[str] = None,
    modality: Optional[str] = None,
    automatable_only: bool = False,
) -> List[DataSource]:
    """Filtra o catálogo por língua, modalidade e capacidade de download."""
    return [
        source
        for source in DATA_SOURCES
        if (language is None or source.language == language)
        and (modality is None or source.modality == modality)
        and (not automatable_only or source.automatable)
    ]
