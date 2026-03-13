# Arquitetura e Fluxos - LibrIA

Este documento organiza o projeto em diagramas pequenos, cada um cobrindo uma etapa clara do fluxo. A ideia e permitir leitura incremental: quando uma etapa termina, a proxima navegacao fica indicada logo abaixo do diagrama.

## Como navegar

1. Comece pela visao geral.
2. Siga para a etapa do fluxo que voce quer entender.
3. Use os links abaixo de cada diagrama para continuar a leitura sem precisar voltar ao topo.

## 1. Visao geral do projeto

```mermaid
flowchart TD
    A[main.py ou Makefile] --> B[Coleta e calibracao]
    A --> C[Treinamento host]
    A --> D[Pipeline embedded]
    B --> E[dataset/static]
    B --> F[dataset/temporal]
    C --> G[Random Forest + LSTM]
    D --> H[CNN estatica + CNN temporal]
    H --> I[Bundle + pacote Pico]
```

Sequencia desta etapa:
- Proximo: [2. Entradas e Comandos](#2-entradas-e-comandos)
- Se quiser focar em dados: [3. Coleta e Geracao de Dataset](#3-coleta-e-geracao-de-dataset)
- Se quiser focar em embedded: [6. Pipeline Embedded e Export para Pico](#6-pipeline-embedded-e-export-para-pico)

Leitura complementar:
- [../README.md](../README.md)
- [DEVELOPMENT.md](DEVELOPMENT.md)

## 2. Entradas e Comandos

```mermaid
flowchart LR
    A[Usuario] --> B[Makefile]
    A --> C[main.py]
    B --> D[collect-static]
    B --> E[collect-temporal]
    B --> F[train]
    B --> G[train-lstm]
    B --> H[train-embedded-all]
    B --> I[export-embedded]
    B --> J[infer-embedded]
    C --> D
    C --> E
    C --> F
    C --> G
    C --> H
    C --> I
    C --> J
```

Sequencia desta etapa:
- Voltar: [1. Visao geral do projeto](#1-visao-geral-do-projeto)
- Proximo: [3. Coleta e Geracao de Dataset](#3-coleta-e-geracao-de-dataset)
- Pular para inferencia: [5. Inferencia Host](#5-inferencia-host)

Leitura complementar:
- [../README.md](../README.md)
- [DEVELOPMENT.md](DEVELOPMENT.md)

## 3. Coleta e Geracao de Dataset

```mermaid
flowchart TD
    A[Webcam] --> B[Coleta host]
    B --> C[Calibracao opcional]
    C --> D[Extracao de landmarks]
    D --> E[dataset/static/<label>/sample_XXX.npy]
    D --> F[dataset/temporal/<label>/seq_XXX.npy]
    B --> G[frame_XXX.png para referencia visual]
```

Sequencia desta etapa:
- Voltar: [2. Entradas e Comandos](#2-entradas-e-comandos)
- Proximo: [4. Treinamento Host](#4-treinamento-host)
- Se o foco for formatos: [DATASETS.md](DATASETS.md)

Leitura complementar:
- [DATASETS.md](DATASETS.md)
- [DEVELOPMENT.md](DEVELOPMENT.md)

## 4. Treinamento Host

```mermaid
flowchart LR
    A[dataset/static] --> B[LibrasModelTrainer]
    B --> C[model/model.pickle]
    D[dataset/temporal] --> E[LibrasLSTMTrainer]
    E --> F[model/libras_lstm.keras]
    E --> G[model/libras_lstm_labels.pickle]
```

Sequencia desta etapa:
- Voltar: [3. Coleta e Geracao de Dataset](#3-coleta-e-geracao-de-dataset)
- Proximo: [5. Inferencia Host](#5-inferencia-host)
- Se o foco for embedded: [6. Pipeline Embedded e Export para Pico](#6-pipeline-embedded-e-export-para-pico)

Leitura complementar:
- [DEVELOPMENT.md](DEVELOPMENT.md)
- [DATASETS.md](DATASETS.md)

## 5. Inferencia Host

```mermaid
flowchart TD
    A[Webcam] --> B[Inferencia estatica]
    A --> C[Janela temporal]
    B --> D[LibrasRealtimeClassifier]
    C --> E[LibrasLSTMRealtimeClassifier]
    D --> F[Predicao estatica]
    E --> G[Predicao temporal]
    F --> H[Arbitragem hibrida]
    G --> H
    H --> I[Overlay e resultado final]
```

Sequencia desta etapa:
- Voltar: [4. Treinamento Host](#4-treinamento-host)
- Proximo: [6. Pipeline Embedded e Export para Pico](#6-pipeline-embedded-e-export-para-pico)
- Pular para runtime final: [7. Runtime no Dispositivo](#7-runtime-no-dispositivo)

Leitura complementar:
- [../README.md](../README.md)
- [DEVELOPMENT.md](DEVELOPMENT.md)

## 6. Pipeline Embedded e Export para Pico

```mermaid
flowchart TD
    A[dataset/static sample_XXX.npy] --> B[LibrasEmbeddedCNNTrainer]
    C[dataset/temporal seq_XXX.npy] --> D[LibrasEmbeddedTemporalCNNTrainer]
    B --> E[TFLite int8 estatico]
    D --> F[TFLite int8 temporal]
    E --> G[build_embedded_bundle]
    F --> G
    G --> H[embedded_bundle.json]
    G --> I[header de configuracao]
    G --> J[pico_package/]
    G --> K[zip para exportacao]
```

Sequencia desta etapa:
- Voltar: [5. Inferencia Host](#5-inferencia-host)
- Proximo: [7. Runtime no Dispositivo](#7-runtime-no-dispositivo)
- Validacao no host: [8. Verificacao do Bundle Embedded](#8-verificacao-do-bundle-embedded)

Leitura complementar:
- [DATASETS.md](DATASETS.md)
- [../README.md](../README.md)

## 7. Runtime no Dispositivo

```mermaid
flowchart LR
    A[Extrator local de ROI/landmarks] --> B[Tensor estatico 21x3]
    A --> C[Janela temporal 30x63]
    B --> D[LibriaEmbeddedRuntime::PredictStatic]
    C --> E[LibriaEmbeddedRuntime::PredictTemporal]
    D --> F[Arbitragem hibrida]
    E --> F
    F --> G[Token final e confianca]
```

Sequencia desta etapa:
- Voltar: [6. Pipeline Embedded e Export para Pico](#6-pipeline-embedded-e-export-para-pico)
- Proximo: [8. Verificacao do Bundle Embedded](#8-verificacao-do-bundle-embedded)
- Se quiser detalhes dos artefatos: [DATASETS.md](DATASETS.md)

Leitura complementar:
- [../src/interfaces/libria_embedded_runtime.h](../src/interfaces/libria_embedded_runtime.h)
- [../src/interfaces/libria_embedded_runtime.cpp](../src/interfaces/libria_embedded_runtime.cpp)

## 8. Verificacao do Bundle Embedded

```mermaid
flowchart TD
    A[embedded_bundle.json] --> B[LibrasEmbeddedRuntime Python]
    C[dataset/static] --> B
    D[dataset/temporal] --> B
    B --> E[Acuracia estatica]
    B --> F[Acuracia temporal]
    B --> G[Acuracia hibrida]
    B --> H[Contrato de landmarks validado]
```

Sequencia desta etapa:
- Voltar: [7. Runtime no Dispositivo](#7-runtime-no-dispositivo)
- Reiniciar leitura: [1. Visao geral do projeto](#1-visao-geral-do-projeto)
- Operacao pratica: [../README.md](../README.md)

Leitura complementar:
- [../README.md](../README.md)
- [DEVELOPMENT.md](DEVELOPMENT.md)

## Resumo de navegacao

- Inicio rapido: [1. Visao geral do projeto](#1-visao-geral-do-projeto)
- Fluxo de dados: [3. Coleta e Geracao de Dataset](#3-coleta-e-geracao-de-dataset)
- Fluxo host: [4. Treinamento Host](#4-treinamento-host)
- Fluxo embedded: [6. Pipeline Embedded e Export para Pico](#6-pipeline-embedded-e-export-para-pico)
- Runtime final: [7. Runtime no Dispositivo](#7-runtime-no-dispositivo)

Ultima atualizacao: 2026-03-12