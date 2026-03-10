# LibrIA - 3 Camadas de Robustez

Guia de acompanhamento da evolucao do projeto com base no estado atual do repositorio.

## Objetivo

As tres camadas abaixo representam a estrategia de robustez do LibrIA para sair de um pipeline estatico basico e chegar a um fluxo mais resiliente para uso real.

## Estado atual resumido

Hoje o projeto ja possui implementacoes para as tres camadas:

1. Features configuraveis por `FEATURE_MODE` em `config/settings.py`.
2. Calibracao opcional de camera com persistencia em `config/*.npy`.
3. Pipeline temporal com coleta de sequencias, treino LSTM e inferencia em tempo real.

A diferenca entre as camadas agora nao e mais "ideia vs implementacao", e sim o quanto cada uma esta consolidada e validada com dados reais.

## Camada 1 - Features robustas

### O que existe hoje

- `FEATURE_DIMENSIONS` em `config/settings.py`
- `FEATURE_MODE = 'wrist_relative'` como padrao atual
- `extract_landmarks_by_mode()` em `utils/helpers.py`
- Persistencia de `feature_mode` no dataset processado e no modelo Random Forest

### Modos suportados

| Modo | Dimensao | Uso |
|---|---|---|
| `bounding_box` | 42 | compatibilidade com pipeline antigo |
| `wrist_relative` | 63 | padrao atual para treino e inferencia |

### Ganho esperado

- menor sensibilidade a escala e posicao da mao
- consistencia entre treino e inferencia
- melhor compatibilidade com modelos sequenciais

### Risco remanescente

Ainda falta medir com mais rigor o impacto comparativo entre `bounding_box` e `wrist_relative` em datasets reais do projeto.

## Camada 2 - Calibracao de camera

### O que existe hoje

- `scripts/generate_checkerboard.py`
- `scripts/show_checkerboard.py`
- `scripts/calibrate_camera.py`
- `load_camera_calibration()` e `preprocess_frame()` em `utils/helpers.py`
- integracao com inferencia estatica e temporal

### Fluxo atual

```bash
make generate-checkerboard
make show-checkerboard
make capture-calibration
```

Arquivos gerados:

- `config/camera_matrix.npy`
- `config/dist_coeffs.npy`

### Ganho esperado

- menor distorcao optica antes da extracao de landmarks
- melhor reproducibilidade entre cameras diferentes
- base melhor para sequencias temporais

### Risco remanescente

A calibracao ainda depende de disciplina operacional: boa captura do tabuleiro, multiplos angulos e controle de qualidade das imagens.

## Camada 3 - Modelagem temporal

### O que existe hoje

- `scripts/collect_sequences.py`
- `src/model_training/libras_lstm_trainer.py`
- `src/inference/libras_lstm_realtime_classifier.py`
- comandos `make collect-sequences`, `make train-lstm`, `make infer-lstm` e `make run-lstm`

### Estrutura atual

- sequencias salvas em `dataset/sequences/<label>/seq_XXX.npy`
- shape esperado por padrao: `(30, 63)`
- modelo salvo em `model/libras_lstm.keras`
- labels e metadados salvos em `model/libras_lstm_labels.pickle`

### Ganho esperado

- representacao melhor de sinais dinamicos como J e Z
- buffer temporal em inferencia em vez de classificacao isolada por frame
- base reutilizavel para futuros modelos Transformer

### Risco remanescente

O pipeline temporal ainda precisa de mais dados rotulados e mais avaliacao sistematica antes de ser tratado como substituto do fluxo classico.

## Compatibilidade de dados

| Camada | Reaproveita dados atuais | Requer coleta nova |
|---|---|---|
| 1 - Features robustas | Sim | Nao, apenas retreino |
| 2 - Calibracao | Sim | Apenas fotos do tabuleiro |
| 3 - Temporal | Parcialmente | Sim, sequencias de video |

## Prioridade recomendada de consolidacao

1. Padronizar o `FEATURE_MODE` usado nos experimentos e registrar resultados.
2. Criar rotina de verificacao da calibracao antes de sessoes de coleta/inferencia.
3. Ampliar o dataset temporal para alem de J e Z e registrar metricas por classe.
4. Comparar Random Forest, LSTM e futuros modelos Transformer com o mesmo protocolo.

## Relacao com o Transformer experimental

O diretorio `src/models/transformer-gpt/` continua sendo uma trilha experimental. A infraestrutura criada nas tres camadas, principalmente `wrist_relative`, calibracao opcional e sequencias em `.npy`, ja prepara o terreno para essa evolucao.

## Status do plano

- Camada 1: implementada
- Camada 2: implementada
- Camada 3: implementada com validacao ainda parcial

Atualizado em 2026-03-10.
