# Datasets e Artefatos - LibrIA

Este documento descreve o dataset unificado, os formatos `NPY` usados no treino e os artefatos gerados pelos fluxos host e embedded.

## Navegação rápida

- Visão dos fluxos em diagramas: [ARCHITECTURE.md](ARCHITECTURE.md)
- Setup e execução local: [DEVELOPMENT.md](DEVELOPMENT.md)

## Visão geral

O projeto hoje usa um dataset unificado com dois subconjuntos principais:

1. Fluxo estático host e embedded: landmarks em `sample_XXX.npy` por classe.
2. Fluxo temporal host e embedded: sequências de landmarks em `seq_XXX.npy` por classe.

## 1. Dataset estático unificado

### Coleta local

```bash
make collect-static
```

Estrutura gerada:

```text
dataset/static/
├── A/
│   ├── frame_000.png
│   ├── sample_000.npy
│   └── ...
├── B/
└── ...
```

Cada pasta representa uma classe estática. O conjunto mínimo recomendado é composto por 24 letras: `A-H`, `I` e `K-Y`.

Os sinais `J` e `Z` ficam fora do fluxo estático mínimo e devem ser coletados no fluxo temporal.

Cada `sample_XXX.npy` contém os landmarks normalizados da mão. No modo padrão `wrist_relative`, o arquivo é salvo com shape:

```text
(21, 3)
```

Uso por pipeline:

- Random Forest host: achatamento para 63 features.
- CNN embedded estática: tensor mantido como `(21, 3)`.

### Dataset externo de apoio

O repositório mantém arquivos de apoio em `data/archives/`, incluindo o dataset ASL organizado em:

```text
data/archives/
├── asl_alphabet_train/
└── asl_alphabet_test/
```

Esses dados são auxiliares e não fazem parte do fluxo principal documentado.

## 2. Dataset temporal unificado

Coleta recomendada:

```bash
make collect-temporal SEQUENCE_LABELS=J\ Z SEQUENCE_COUNT=30 SEQUENCE_LENGTH=30
```

Estrutura gerada:

```text
dataset/temporal/
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
(sequence_length, 21, 3)
```

No padrão atual isso significa:

```text
(30, 21, 3)
```

Uso por pipeline:

- LSTM host: reshape interno conforme o trainer temporal.
- CNN embedded temporal: reshape para `(30, 63)` quando necessario.

## 3. Modos de feature

O valor de `num_features` depende de `FEATURE_MODE` em `config/settings.py`:

- `bounding_box`: 42 features
- `wrist_relative`: 63 features

O modo padrão atual é `wrist_relative`.

## 4. Manifestos e metadados

Cada subconjunto pode manter um `manifest.json` com:

- `mode`
- `feature_mode`
- `feature_dimension`
- `sample_target`
- `sequence_length`
- `camera_calibrated`
- `counts`

Esse arquivo é auxiliar e não é a fonte principal de treino.

## 5. Artefatos de modelos

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

### Modelos embedded

```text
model/libria_embedded_cnn.keras
model/libria_embedded_cnn_int8.tflite
model/libria_embedded_cnn_labels.json
model/libria_embedded_temporal_cnn.keras
model/libria_embedded_temporal_cnn_int8.tflite
model/libria_embedded_temporal_cnn_labels.json
```

Execução:

```bash
make train-embedded-all
```

### Bundle embedded e pacote Pico

```text
model/embedded_bundle/
├── embedded_bundle.json
├── libria_embedded_bundle_config.h
├── libria_embedded_cnn_int8.tflite
├── libria_embedded_temporal_cnn_int8.tflite
└── pico_package/
    ├── include/
    ├── src/
    ├── examples/
    ├── CMakeLists.txt
    └── README.md
```

Execução:

```bash
make export-embedded
make infer-embedded
```

O pacote `pico_package/` ja sai preparado para exportacao ao RP2040/Pico e inclui arrays C/C++ com os modelos quantizados, manifesto, header de configuracao e runtime skeleton.

## 6. Calibração de câmera

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

## 7. Modelos e pesos adicionais já versionados

Na pasta `model/` e em áreas legadas do projeto existem artefatos de experimentos anteriores, incluindo:

- `best_temporal_model.h5`
- `temporal_cnn_model.h5`
- `libras_lstm.keras`
- `asl_vgg16_best_weights.keras`

Esses arquivos não substituem automaticamente o fluxo principal documentado. Use-os apenas se o experimento correspondente estiver configurado no código.

## 8. Relação com MediaPipe e runtime embedded

- O MediaPipe continua no host como extrator de landmarks durante coleta e fluxos de inferência host.
- O dataset salvo em `NPY` vira o contrato entre coleta e treino.
- O bundle embedded nao depende de MediaPipe em runtime.
- No dispositivo, o firmware precisa apenas reproduzir o mesmo layout de landmarks normalizados e o mesmo ROI controlado.

## 9. Checklist rápido

- [ ] Escolhi o fluxo: estático ou temporal
- [ ] Tenho dados em `dataset/static/` ou em `dataset/temporal/`
- [ ] Sei qual `FEATURE_MODE` está ativo
- [ ] Rodei `make verify-setup`
- [ ] Treinei o modelo correspondente antes de inferir

Ultima atualizacao: 2026-03-12
