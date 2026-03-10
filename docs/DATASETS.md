# Datasets e Artefatos - LibrIA

Este documento descreve os formatos de dados usados pelo LibrIA e onde cada artefato entra no pipeline.

## Visão geral

O projeto hoje suporta dois fluxos principais:

1. Fluxo estático: imagens por classe, processamento com MediaPipe e treino de Random Forest.
2. Fluxo temporal: sequências de landmarks por classe, treino de LSTM e inferência com janela deslizante.

## 1. Dados brutos para o fluxo estático

### Coleta local

```bash
make collect
```

Ou:

```bash
python main.py collect
python main.py collect_jz
```

Estrutura gerada:

```text
data/
├── 0/
├── 1/
├── ...
└── 25/
```

Cada pasta representa uma classe do alfabeto (`0 = A`, `25 = Z`).

### Dataset externo de apoio

O repositório mantém arquivos de apoio em `data/archives/`, incluindo o dataset ASL organizado em:

```text
data/archives/
├── asl_alphabet_train/
└── asl_alphabet_test/
```

Quando usar:

- para testes rápidos de pipeline
- para comparar distribuição de dados
- para experimentos com modelos alternativos

## 2. Dataset processado do fluxo estático

Após o processamento:

```bash
make process
```

é gerado o arquivo:

```text
dataset/data.pickle
```

Campos principais:

- `data`: vetores de features por amostra
- `labels`: classes associadas
- `num_features`: dimensionalidade efetiva
- `num_classes`: número de classes
- `feature_mode`: modo usado na extração

### Modos de feature

O valor de `num_features` depende de `FEATURE_MODE` em `config/settings.py`:

- `bounding_box`: 42 features
- `wrist_relative`: 63 features

O modo padrão atual é `wrist_relative`.

## 3. Dados brutos do fluxo temporal

Coleta recomendada:

```bash
make collect-sequences SEQUENCE_LABELS=J\ Z SEQUENCE_COUNT=30 SEQUENCE_LENGTH=30
```

Estrutura gerada:

```text
dataset/sequences/
├── J/
│   ├── seq_000.npy
│   ├── seq_001.npy
│   └── ...
└── Z/
    ├── seq_000.npy
    ├── seq_001.npy
    └── ...
```

Cada arquivo `.npy` contém uma sequência com shape:

```text
(sequence_length, feature_dimension)
```

No padrão atual isso significa:

```text
(30, 63)
```

## 4. Artefatos de modelos

### Modelo clássico

```text
model/model.pickle
```

Conteúdo esperado:

- classificador Random Forest
- histórico de treino
- `feature_mode`
- `num_features`

Execução:

```bash
make infer
```

### Modelo temporal

```text
model/libras_lstm.keras
model/libras_lstm_labels.pickle
```

Execução:

```bash
make train-lstm
make infer-lstm
```

## 5. Calibração de câmera

Arquivos gerados:

```text
config/camera_matrix.npy
config/dist_coeffs.npy
```

Esses arquivos são opcionais, mas quando presentes podem ser usados para corrigir distorção antes da extração dos landmarks.

Fluxo recomendado:

```bash
make generate-checkerboard
make show-checkerboard
make capture-calibration
```

## 6. Modelos e pesos adicionais já versionados

Na pasta `model/` e em áreas legadas do projeto existem artefatos de experimentos anteriores, incluindo:

- `best_temporal_model.h5`
- `temporal_cnn_model.h5`
- `libras_lstm.keras`
- `asl_vgg16_best_weights.keras`

Esses arquivos não substituem automaticamente o fluxo principal documentado. Use-os apenas se o experimento correspondente estiver configurado no código.

## 7. Checklist rápido

- [ ] Escolhi o fluxo: estático ou temporal
- [ ] Tenho dados em `data/` ou em `dataset/sequences/`
- [ ] Sei qual `FEATURE_MODE` está ativo
- [ ] Rodei `make verify-setup`
- [ ] Treinei o modelo correspondente antes de inferir

Última atualização: 2026-03-10
