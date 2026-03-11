# Documentação de Desenvolvimento

Guia de setup e rotina de trabalho para quem vai mexer no código do LibrIA.

## Pré-requisitos

- Python 3.11+
- Git
- Webcam para fluxos de coleta e inferência
- CPU com suporte AVX para MediaPipe, TensorFlow e boa parte do stack temporal

## Setup recomendado

### Com Makefile

```bash
git clone https://github.com/Gabryel-lima/LibrIA.git
cd LibrIA
make setup
source .venv/bin/activate
make verify-setup
```

### Manual

```bash
git clone https://github.com/Gabryel-lima/LibrIA.git
cd LibrIA
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python test_setup.py
```

## Dependências de desenvolvimento

`requirements-dev.txt` inclui:

- `pytest`, `pytest-cov`, `pytest-xdist`
- `black`, `flake8`, `isort`, `mypy`, `pylint`
- `pre-commit`
- `sphinx`
- `ipython`, `jupyter`, `notebook`

## Comandos de rotina

```bash
make test
make lint
make format
make environment
```

Ou diretamente:

```bash
python test_setup.py
pytest tests/ -v
black src/ main.py
flake8 src/ main.py --max-line-length=119 --exclude=__pycache__
```

## Fluxos do projeto para desenvolvimento

### Pipeline estático

```bash
make collect-static
make train
make infer
```

### Pipeline temporal

```bash
make collect-temporal SEQUENCE_LABELS=J\ Z SEQUENCE_COUNT=30 SEQUENCE_LENGTH=30
make train-lstm
make infer-lstm
```

### Pipeline mínimo recomendado

```bash
make collect-minimal-dataset
make train-hybrid
make infer-hybrid
```

### Calibração de câmera

```bash
make generate-checkerboard
make show-checkerboard
make capture-calibration
```

## Estrutura relevante para desenvolvimento

```text
src/
├── inference/
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
- `FEATURE_DIMENSIONS`: dimensionalidade por modo
- `CAMERA_CONFIG`: calibração opcional
- `LSTM_CONFIG`: sequência, batch size, épocas e caminhos dos artefatos

## Convenções úteis

- Use `main.py` ou o Makefile para manter o mesmo fluxo documentado
- O dataset estático fica em `dataset/static/<label>/sample_XXX.npy`
- O dataset temporal fica em `dataset/temporal/<label>/seq_XXX.npy`
- O modelo clássico fica em `model/model.pickle`
- O modelo temporal fica em `model/libras_lstm.keras`

## Problemas comuns

### `illegal hardware instruction`

Consulte [AVX_COMPATIBILITY.md](AVX_COMPATIBILITY.md). Em CPUs sem AVX, MediaPipe, TensorFlow e algumas partes do PyTorch podem falhar já no import.

### `Modelo LSTM não encontrado`

Rode primeiro:

```bash
make collect-temporal
make train-lstm
```

### `Nenhuma sequência válida encontrada`

Verifique se os `.npy` em `dataset/temporal/` têm shape compatível com `LSTM_CONFIG['sequence_length']` e `FEATURE_DIMENSION`.

Última atualização: 2026-03-10
```bash
# Use cache aggressivo
pip install --cache-dir /tmp/pip-cache -r requirements.txt

# Ou use mirror brasileiro
pip install -i https://pypi.tsinghua.edu.cn/simple -r requirements.txt
```

### Testes falhando localmente mas passando no CI

**Solução:**
```bash
# Limpe cache pytest
pytest --cache-clear tests/

# Reinstale dependências
pip install --force-reinstall -r requirements-dev.txt
```

## 📖 Documentação Local

### Buildar Sphinx docs

```bash
cd docs

# Limpar builds anteriores
make clean

# Build HTML
make html

# Abrir no navegador
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
start _build/html/index.html  # Windows
```

## 🔐 Environment Variables

Criar `.env` baseado em `.env.example`:

```bash
cp .env.example .env

# Editar .env com valores locais
export $(cat .env | xargs)
python main.py
```

## 📝 Checklist Antes de Fazer PR

- [ ] Código formatado com Black
- [ ] Imports organizados com isort
- [ ] Pylint score 8.0+
- [ ] mypy type check passou
- [ ] Testes passam: `pytest tests/ -v`
- [ ] Coverage 80%+: `pytest --cov=src`
- [ ] Docstrings em todas funções públicas
- [ ] Sem `print()` statements (use logging)
- [ ] Nenhum `TODO` ou `FIXME` esquecido
- [ ] CHANGELOG.md atualizado
- [ ] Commits bem estruturados e mensagens claras

## 🆘 Precisa de Ajuda?

- 📧 Email: gabbryellimasi@gmail.com
- 💬 Discussions: [Issues & Discussions](https://github.com/Gabryel-lima/LibrIA)
- 📖 Docs: [LibrIA Documentation](https://Gabryel-lima.github.io/LibrIA)

---

Última atualização: 2026-02-16
