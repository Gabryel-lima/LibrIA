# Documentação de Desenvolvimento

Guia de setup, rotina local e pontos de atenção para quem vai mexer no código do LibrIA.

## Navegação rápida

- Visão visual dos fluxos: [ARCHITECTURE.md](ARCHITECTURE.md)
- Estrutura de dados e artefatos: [DATASETS.md](DATASETS.md)
- Limitações de CPU e AVX: [AVX_COMPATIBILITY.md](AVX_COMPATIBILITY.md)

## Pré-requisitos

- Python 3.11+
- Git
- Webcam para fluxos de coleta e inferência
- CPU com suporte AVX para usar MediaPipe e TensorFlow com menos restrições

## Setup recomendado

### Com Makefile

```bash
git clone https://github.com/Gabryel-lima/LibrIA.git
cd LibrIA
make setup
source .venv/bin/activate
make verify
```

### Manual

```bash
git clone https://github.com/Gabryel-lima/LibrIA.git
cd LibrIA
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -m unittest discover -s tests -t .
```

## Comandos de rotina

```bash
make test
make lint
make format
make environment
```

Ou diretamente:

```bash
python -m unittest discover -s tests -t .          # suite completa
python -m unittest tests.test_temporal_pipeline    # um modulo so
black src/ scripts/ config/ main.py
flake8 src/ scripts/ config/ main.py --max-line-length=119 --exclude=__pycache__
```

Se o comando vier de `main.py`, o processo agora encerra com `0` em sucesso e `1` em falha. Isso permite usar `make train`, `python main.py train` e comandos similares em automações sem parsear texto do terminal.

## Fluxos principais para desenvolvimento

### 1. Pipeline estático host

```bash
make collect-static
make train
make infer
```

Artefatos principais:
- `dataset/static/<label>/sample_XXX.npy`
- `dataset/static/<label>/sample_XXX_mirror.npy`
- `model/model.pickle`

Observação prática:
- A coleta estática faz backfill dos `sample_XXX.npy` ausentes a partir de `frame_XXX.png` já existentes e grava a versão espelhada na mesma passada.

### 2. Pipeline temporal host

```bash
make collect-temporal SUBJECT=ana SEQUENCES=30
make train-temporal
make infer-temporal
```

Artefatos principais:
- `dataset/temporal/<label>/seq_XXX.npy`
- `dataset/temporal/<label>/seq_XXX_mirror.npy`
- `model/libras_lstm.keras`
- `model/libras_lstm_labels.pickle`
- `training_plots/training_history_lstm.png`

Observação prática:
- O trainer LSTM usa `LEGACY_TEMPORAL_DATASET_DIR` como fallback quando o diretório temporal principal configurado existe, mas está vazio.

### 3. Pipeline híbrido host

```bash
make collect
make train
make infer
```

Notas de comportamento:
- `make train` falha com status não nulo se o dataset estático ou temporal estiver ausente, ou se alguma das duas etapas de treino falhar.
- `make infer` testa a hipótese original e a hipótese espelhada nas trilhas estática e temporal antes da arbitragem final.

### 4. Pipeline embedded

```bash
make embedded-train
make embedded-export
make embedded-check
```

Artefatos principais:
- `model/libria_embedded_cnn_int8.tflite`
- `model/libria_embedded_temporal_cnn_int8.tflite`
- `model/embedded_bundle/embedded_bundle.json`
- `model/embedded_bundle/pico_package/`
- `training_plots/training_history_embedded.png`
- `training_plots/training_history_temporal_embedded.png`
- `training_plots/accuracy/accuracy_*.png`

### 5. Calibração de câmera

```bash
make checkerboard
make checkerboard-show
make calibrate-capture
```

## Estrutura relevante para desenvolvimento

```text
src/
├── inference/
├── interfaces/
└── model_training/

scripts/
├── calibrate_camera.py
├── collect_dataset.py
├── generate_checkerboard.py
└── show_checkerboard.py

config/
└── settings.py
```

## Configurações importantes

As principais chaves ficam em `config/settings.py`:

- `FEATURE_MODE`: `bounding_box` ou `wrist_relative`
- `CAMERA_CONFIG`: calibração opcional
- `LSTM_CONFIG`: sequência, batch size e caminhos do fluxo temporal host
- `EMBEDDED_CONFIG`: treino estático quantizado
- `EMBEDDED_TEMPORAL_CONFIG`: treino temporal quantizado
- `EMBEDDED_BUNDLE_CONFIG`: bundle final e pacote do Pico

## Convenções úteis

- Use `main.py` ou o Makefile para manter o mesmo fluxo documentado.
- O dataset estático fica em `dataset/static/<label>/sample_XXX.npy`.
- O dataset temporal fica em `dataset/temporal/<label>/seq_XXX.npy`.
- Sufixos `_mirror.npy` fazem parte do pipeline atual e não devem ser removidos do dataset ativo sem intenção explícita.
- O Random Forest fica em `model/model.pickle`.
- O modelo temporal host fica em `model/libras_lstm.keras`.
- O runtime embedded no dispositivo parte de `src/interfaces/libria_embedded_runtime.h` e `src/interfaces/libria_embedded_runtime.cpp`.

## Problemas comuns

### `illegal hardware instruction`

Consulte [AVX_COMPATIBILITY.md](AVX_COMPATIBILITY.md). Em CPUs sem AVX, MediaPipe, TensorFlow e partes do PyTorch podem falhar no import.

### `Modelo LSTM não encontrado`

Rode primeiro:

```bash
make collect-temporal
make train-temporal
```

### `Nenhuma sequência válida encontrada`

Verifique se os `.npy` em `dataset/temporal/` têm shape compatível com `LSTM_CONFIG['sequence_length']` e `FEATURE_DIMENSION`.

### `Bundle embedded não encontrado`

Rode primeiro:

```bash
make embedded-train
```

ou, se os `.tflite` já existirem:

```bash
make embedded-export
```

## Checklist antes de fazer PR

- [ ] Rodei `make verify`
- [ ] Rodei os testes relevantes para a mudança
- [ ] Mantive a documentação alinhada ao fluxo real do código
- [ ] Atualizei `CHANGELOG.md` quando a mudança alterou comportamento visível
- [ ] Revisei se o impacto em host e embedded foi coberto quando aplicável

Ultima atualizacao: 2026-03-17
