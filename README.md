# LibrIA — Reconhecimento e Tradução de Libras

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

[![Vídeo de Demonstração](https://img.shields.io/badge/YouTube-Demonstração-red)](https://youtu.be/WT0BZP_kRvQ?si=gDnGLICMUs_cUyn-)

</div>

## Sobre

O **LibrIA** reconhece Libras a partir da webcam usando landmarks de mão do
MediaPipe. O repositório cobre o ciclo completo: coleta de dataset, treino,
inferência em tempo real e export de um bundle quantizado com pacote C/C++
pronto para o Raspberry Pi Pico.

O sistema reconhece **sinais isolados** — o alfabeto manual e sinais dinâmicos.
A composição em palavras e frases é o trabalho em andamento; veja o
[plano arquitetural](.github/plans/atualização_projeto.md) e as fases já
implementadas em [docs/FASE1_RECONHECIMENTO.md](docs/FASE1_RECONHECIMENTO.md) e
[docs/FASE2_TEMPORAL.md](docs/FASE2_TEMPORAL.md).

## Início rápido

```bash
git clone https://github.com/Gabryel-lima/LibrIA.git
cd LibrIA

make setup                      # cria o venv e instala as dependências
make verify                     # confere se o ambiente está utilizável

make report                     # o que já existe e o que falta por classe
make sources                    # bases públicas de Libras que cobrem as lacunas

# Ingestão de base externa — cobre classes inteiras sem gravar nada
make ingest SOURCE_DIR=data/archives/minds-libras MODALITY=temporal

# Coleta — só o que sobrou, e sempre identifique quem está sinalizando
make collect SUBJECT=ana ENVIRONMENT=sala_luz_natural CAMERA_ID=c920 DOMINANT_HAND=right

make train                      # treina os modelos estático e temporal
make infer                      # inferência híbrida em tempo real ('q' para sair)
```

`make help` lista todos os comandos. Pré-requisitos: Python 3.11+, webcam e uma
CPU com suporte a AVX (necessário para o MediaPipe — veja
[docs/AVX_COMPATIBILITY.md](docs/AVX_COMPATIBILITY.md) se a sua não tiver).

## Comandos

Cada alvo do Makefile tem um comando de mesmo nome no `main.py`:
`make train-temporal` é idêntico a `python main.py train-temporal`. Use o
Makefile no dia a dia; o `main.py` existe para quem não usa `make`.

### Pipeline

| Comando | O que faz |
|:--------|:----------|
| `make sources` | Catálogo de bases públicas de Libras que podem alimentar o dataset |
| `make fetch SOURCE=<chave>` | Baixa uma base com download automatizável |
| `make ingest SOURCE_DIR=<dir>` | Converte a base baixada em amostras do dataset (sem webcam) |
| `make collect` | Dataset mínimo: alfabeto estático (24 letras) + J/Z |
| `make collect-static` | Só o alfabeto manual estático |
| `make collect-temporal` | Só as letras temporais (J, Z) |
| `make collect-words` | Palavras e gestos funcionais do vocabulário lexical |
| `make collect-unknown` | Amostras fora do vocabulário (classe de rejeição) |
| `make report` | Cobertura do dataset: vocabulário, metadados, divisão por pessoa |
| `make train` | Treina os modelos estático e temporal |
| `make train-static` | Só o Random Forest estático |
| `make train-temporal` | Só a LSTM temporal |
| `make infer` | **Inferência híbrida** (recomendado) |
| `make infer-static` | Só o modelo estático |
| `make infer-temporal` | Só o modelo temporal |
| `make all` | Pipeline completo: coletar → treinar → inferir |

### Variáveis de coleta

| Variável | Padrão | Para quê |
|:---------|:-------|:---------|
| `SUBJECT` | `desconhecido` | Quem está sinalizando — **sem isso não há divisão por pessoa** |
| `CAMERA_ID` | `desconhecido` | Modelo/apelido da câmera |
| `ENVIRONMENT` | `desconhecido` | Ambiente da captura (ex.: `sala_luz_natural`) |
| `DOMINANT_HAND` | `desconhecido` | `left` ou `right` |
| `CAMERA` | `0` | Índice da webcam |
| `SAMPLES` | `30` | Amostras estáticas por classe |
| `SEQUENCES` | `30` | Sequências temporais por classe |

```bash
make collect-words SUBJECT=bruno ENVIRONMENT=escritorio SEQUENCES=40
```

A coleta é dirigida por lacunas: classes que já atingiram a meta (`SAMPLES` /
`SEQUENCES`) são puladas, venham elas da webcam ou de uma base externa. O plano
é impresso antes de a câmera abrir. Para regravar mesmo assim, use
`python main.py collect-static --all-labels`.

### Variáveis de dados externos

| Variável | Para quê |
|:---------|:---------|
| `SOURCE` | Chave da base em `make fetch` (ex.: `minds-libras`) |
| `SOURCE_DIR` | Diretório da base já baixada, uma pasta por sinal |
| `SOURCE_NAME` | Vira `source_dataset` nos metadados de cada amostra |
| `MODALITY` | `temporal` (padrão) ou `static` |
| `LABEL_MAP` | JSON `{termo_da_base: LABEL_LIBRIA}` para casar o vocabulário |

Detalhes e as bases catalogadas em [docs/DATASETS.md](docs/DATASETS.md#9-dados-externos-sem-coleta-manual).

### Setup, embedded, câmera e desenvolvimento

| Comando | O que faz |
|:--------|:----------|
| `make setup` / `make verify` | Cria o ambiente / valida as dependências |
| `make install-cpu` / `make install-gpu` | Instala dependências (CUDA detectado automaticamente) |
| `make embedded-train` | Treina as CNNs quantizadas INT8 e exporta o bundle |
| `make embedded-export` | Empacota modelos, manifesto e pacote C/C++ do Pico |
| `make embedded-check` | Valida o bundle sobre os datasets `.npy` |
| `make checkerboard` / `make calibrate-capture` / `make calibrate` | Calibração de câmera com tabuleiro 9x6 |
| `make test` | Roda a suíte de testes |
| `make lint` / `make format` | flake8 / black (requer `make install-dev`) |
| `make clean` / `make clean-all` | Limpa caches / remove venv, dataset e modelos |

## Como funciona

### As quatro camadas

1. **Percepção visual** — webcam → MediaPipe → landmarks normalizados.
2. **Reconhecimento de sinais** — classificação estática e temporal, com
   detecção de início e fim de cada sinal.
3. **Composição linguística** — tokens → palavras → português. *(em aberto)*
4. **Apresentação** — texto em tempo real, histórico e confiança.

### Pipeline de reconhecimento

```
webcam → MediaPipe → features → buffer → movimento → segmentação
                                   ↓                      ↓
                          fallback estático        modelo temporal
                                   ↓                      ↓
                                   └──→ suavização → dedup → SignToken
```

O modelo temporal só é consultado quando há um sinal delimitado por movimento;
o modelo estático responde enquanto a mão está parada formando uma letra.
Detalhes em [docs/FASE2_TEMPORAL.md](docs/FASE2_TEMPORAL.md).

Toda saída é um `SignToken` padronizado:

```python
{
  'label': 'OI', 'token': 'oi', 'confidence': 0.91,
  'start_time': 12.4, 'end_time': 13.1, 'duration_seconds': 0.7,
  'source': 'temporal',    # temporal | static
  'state': 'final',        # partial | final | rejected
  'sign_type': 'lexical',  # alphabet | lexical | functional
}
```

`state: rejected` é o que evita apresentar tradução errada como certeza: abaixo
do limiar de confiança, o sinal vira `DESCONHECIDO` em vez de virar palavra.

### Modelos

| Modelo | Papel | Artefato |
|:-------|:------|:---------|
| Random Forest | Letras estáticas | `model/model.pickle` |
| LSTM | Sinais dinâmicos (30 quadros × 63 features) | `model/libras_lstm.keras` |
| CNN estática INT8 | Deployment embarcado | `model/libria_embedded_cnn_int8.tflite` |
| CNN temporal INT8 | Deployment embarcado | `model/libria_embedded_temporal_cnn_int8.tflite` |

Todos os modelos recebem os mesmos landmarks; o `FEATURE_MODE`
(`wrist_relative`, 63 features, ou `bounding_box`, 42) define o formato.

## Vocabulário e dataset

[`config/vocabulary.py`](config/vocabulary.py) é a fonte única de verdade do
vocabulário, dividido em três famílias:

| Família | Modalidade | Conteúdo |
|:--------|:-----------|:---------|
| `alphabet` | estática (24) + temporal (J, Z) | Alfabeto manual |
| `lexical` | temporal | Palavras (OI, OBRIGADO, AJUDA, …) |
| `functional` | temporal | ESPACO, PAUSA, APAGAR, CONFIRMAR |

Mais `DESCONHECIDO`, a classe explícita de fora do vocabulário.

### Estrutura em disco

```
dataset/
├── static/<LABEL>/     sample_XXX.npy · sample_XXX_mirror.npy
│                       sample_XXX.json · frame_XXX.png
└── temporal/<LABEL>/   seq_XXX.npy · seq_XXX_mirror.npy · seq_XXX.json
```

Cada `.npy` tem um `.json` irmão com pessoa, câmera, ambiente, mão dominante,
duração e qualidade da captura. São esses metadados que permitem dividir
treino/validação/teste **por pessoa** — dividir por amostra faz a mesma pessoa
aparecer nos três conjuntos e infla a acurácia. Detalhes em
[docs/FASE1_RECONHECIMENTO.md](docs/FASE1_RECONHECIMENTO.md) e
[docs/DATASETS.md](docs/DATASETS.md).

Toda amostra é gravada junto com sua versão espelhada, o que dá suporte a mão
esquerda e direita sem coletar duas vezes.

## Estrutura do projeto

```
LibrIA/
├── main.py                     CLI (mesmos nomes dos alvos do Makefile)
├── Makefile                    setup, coleta, treino, inferência
├── config/
│   ├── settings.py             configuração central
│   └── vocabulary.py           vocabulário (fonte de verdade)
├── scripts/
│   ├── collect_dataset.py      coleta estática e temporal
│   ├── dataset_report.py       relatório de cobertura
│   ├── calibrate_camera.py     calibração de câmera
│   └── generate_checkerboard.py
├── src/
│   ├── dataset/                metadados por amostra
│   ├── evaluation/             divisão por pessoa e métricas
│   ├── model_training/         Random Forest, LSTM e CNNs quantizadas
│   ├── inference/              pipeline temporal e classificadores
│   ├── models/transformer-gpt/ experimental, ainda não integrado
│   └── utils/
├── utils/helpers.py            landmarks, calibração e logging
├── tests/                      suíte de testes (make test)
├── dataset/ · model/ · output/ · training_plots/
└── docs/
```

## Configuração

Tudo em [`config/settings.py`](config/settings.py):

```python
FEATURE_MODE = 'wrist_relative'          # ou 'bounding_box'

COLLECTION_CONFIG = {'static_samples_per_label': 30, ...}
LSTM_CONFIG        = {'sequence_length': 30, 'confidence_threshold': 0.85, ...}
TEMPORAL_PIPELINE_CONFIG = {             # segmentação e suavização
    'motion_start_threshold': 0.012,
    'motion_end_threshold': 0.006,
    'duplicate_window_seconds': 1.0, ...
}
EVALUATION_CONFIG  = {'split_ratios': {...}, 'rejection_threshold': 0.75, ...}
```

Os limiares de movimento estão em unidades de deslocamento médio de landmark
por quadro e **precisam ser calibrados na sua câmera** — o padrão é um ponto de
partida, não uma medição.

### Ampliar o vocabulário

1. Acrescente entradas em `_LEXICAL_ENTRIES` (`config/vocabulary.py`).
2. Veja se uma base pública já cobre o sinal (`make sources`) e ingira
   (`make ingest ... LABEL_MAP=...`). Só grave com `make collect-words
   SUBJECT=...` o que sobrar.
3. Aponte `LSTM_CONFIG['allowed_classes']` para `TEMPORAL_VOCABULARY_LABELS` e
   deixe `require_all_allowed_classes = False`, para treinar com as classes já
   coletadas em vez de falhar nas que ainda faltam.
4. `make train-temporal`.

## Solução de problemas

| Problema | Solução |
|:---------|:--------|
| `illegal hardware instruction (core dumped)` | CPU sem AVX — veja [docs/AVX_COMPATIBILITY.md](docs/AVX_COMPATIBILITY.md) |
| `Ambiente virtual não encontrado` | `make setup` |
| Webcam não detectada | Confira permissões e tente `make infer CAMERA=1` |
| Baixa acurácia | Colete mais pessoas e ambientes; confira `make report` |
| Sinal não é detectado | Ajuste `motion_start_threshold` (veja a seção de configuração) |
| Um sinal vira vários tokens repetidos | Aumente `duplicate_window_seconds` |
| Módulo não encontrado | Rode os comandos a partir da raiz do projeto |

Logs em `libras.log` (configurável em `LOGGING_CONFIG`).

## Documentação

- [docs/FASE1_RECONHECIMENTO.md](docs/FASE1_RECONHECIMENTO.md) — vocabulário, metadados, divisão por pessoa e métricas
- [docs/FASE2_TEMPORAL.md](docs/FASE2_TEMPORAL.md) — pipeline temporal, segmentação e `SignToken`
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — arquitetura e diagramas
- [docs/DATASETS.md](docs/DATASETS.md) — formatos, artefatos e datasets externos
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — setup de desenvolvimento
- [docs/AVX_COMPATIBILITY.md](docs/AVX_COMPATIBILITY.md) — CPUs sem AVX
- [docs/latex/model_architecture_equations.pdf](docs/latex/model_architecture_equations.pdf) — fórmulas da arquitetura

**Comunidade:** [CONTRIBUTING.md](CONTRIBUTING.md) ·
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) · [GOVERNANCE.md](GOVERNANCE.md) ·
[ROADMAP.md](ROADMAP.md) · [CHANGELOG.md](CHANGELOG.md) ·
[CONTRIBUTORS.md](CONTRIBUTORS.md)

## Contribuindo

Contribuições são bem-vindas. Leia o [CONTRIBUTING.md](CONTRIBUTING.md) e o
[guia de Pull Requests](docs/PULL_REQUEST_GUIDE.md). Áreas com maior impacto
agora:

- **Coleta com mais pessoas** — o gargalo real do projeto hoje
- Vocabulário lexical de Libras (palavras e variações regionais)
- Camada de composição linguística (tokens → português)
- Validação com pessoas surdas e intérpretes

## Licença

MIT — veja [LICENSE](LICENSE).

## Autor

**Gabryel Lima** — [GitHub](https://github.com/Gabryel-lima)

Construído com [MediaPipe](https://mediapipe.dev/),
[OpenCV](https://opencv.org/), [scikit-learn](https://scikit-learn.org/) e
[TensorFlow](https://www.tensorflow.org/).
