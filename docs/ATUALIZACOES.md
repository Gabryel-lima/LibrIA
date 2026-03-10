# Atualizações Recentes - LibrIA

## Data

2026-03-10

## Escopo desta atualização

Sincronização da documentação com as mudanças já presentes no código e no Makefile.

## Principais mudanças documentadas

### Pipeline temporal
- Inclusão do fluxo de coleta de sequências em `dataset/sequences/`
- Documentação de `make collect-sequences`, `make train-lstm`, `make infer-lstm` e `make run-lstm`
- Registro do artefato `model/libras_lstm.keras` e do mapa `model/libras_lstm_labels.pickle`

### Extração de features configurável
- Atualização para o uso de `FEATURE_MODE`
- Documentação dos modos `bounding_box` e `wrist_relative`
- Ajuste da documentação de dimensionalidade: 42 ou 63 features, dependendo da configuração

### Calibração de câmera
- Inclusão do fluxo com tabuleiro 9x6
- Documentação dos comandos `make generate-checkerboard`, `make show-checkerboard`, `make capture-calibration` e `make calibrate-camera`
- Registro dos arquivos `config/camera_matrix.npy` e `config/dist_coeffs.npy`

### Dependências e compatibilidade
- Atualização das versões e do status de TensorFlow, Keras, PyTorch e MediaPipe
- Revisão da documentação para máquinas sem AVX

## Arquivos atualizados nesta rodada

- `README.md`
- `DOCUMENTATION_INDEX.md`
- `docs/README.md`
- `docs/DEVELOPMENT.md`
- `docs/DATASETS.md`
- `docs/AVX_COMPATIBILITY.md`
- `CHANGELOG.md`
- `src/models/transformer-gpt/README.md`

## Resultado esperado

- O README principal passa a refletir os dois pipelines suportados pelo projeto
- Os guias de setup e datasets deixam de depender de comandos e arquivos obsoletos
- A documentação interna deixa de assumir que toda feature extraction usa 42 coordenadas fixas
