# 📚 Documentação de Desenvolvimento

Guia para desenvolvedores que desejam configurar o ambiente de desenvolvimento local.

## 🛠️ Pré-requisitos

- Python 3.11+ instalado
- Git configurado
- Acesso a terminal/CMD
- ~5GB de espaço em disco (para dados + modelos)

## 🚀 Setup Rápido (5 minutos)

### MacOS/Linux

```bash
# 1. Clone o repositório
git clone https://github.com/Gabryel-lima/LibrIA.git
cd LibrIA

# 2. Crie ambifiente virtual
python -m venv venv
source venv/bin/activate

# 3. Instale dependências
pip install -r requirements-dev.txt

# 4. Teste a instalação
python test_setup.py

# 5. Execute testes
pytest tests/ -v
```

### Windows

```bash
# 1. Clone o repositório
git clone https://github.com/Gabryel-lima/LibrIA.git
cd LibrIA

# 2. Crie ambiente virtual
python -m venv venv
venv\Scripts\activate

# 3. Instale dependências
pip install -r requirements-dev.txt

# 4. Teste a instalação
python test_setup.py

# 5. Execute testes
pytest tests/ -v
```

## 📦 Dependências de Desenvolvimento

Arquivo: `requirements-dev.txt`

```
# Dependências base
-r requirements.txt

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-mock==3.12.0

# Code Quality
pylint==3.0.3
black==23.12.1
isort==5.13.2
mypy==1.7.1

# Documentation
sphinx==7.2.6
sphinx-rtd-theme==2.0.0

# Development
ipython==8.18.1
jupyter==1.0.0
```

## 🔧 Configuração Recomendada do VS Code

### Extensions Recomendadas

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.debugpy",
    "ms-liveShare.liveShare",
    "eamodio.gitlens",
    "GitHub.copilot",
    "ms-vscode.makefile-tools"
  ]
}
```

### Settings (settings.json)

```json
{
  "[python]": {
    "editor.defaultFormatter": "ms-python.python",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll.pylint": "explicit",
      "source.fixAll.isort": "explicit"
    }
  },
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "python.formatting.provider": "black",
  "python.linting.pylintArgs": [
    "--disable=too-few-public-methods"
  ]
}
```

## 🧪 Executando Testes

### Todos os testes

```bash
pytest tests/ -v
```

### Com cobertura

```bash
pytest tests/ --cov=src --cov-report=html
# Abrir: htmlcov/index.html
```

### Teste específico

```bash
pytest tests/test_data_processing.py::test_normalize -v
```

### Apenas testes rápidos

```bash
pytest -m "not slow" -v
```

## 🎨 Code Formatting

### Black - Formatação

```bash
# Formatar código
black src/ --line-length 100

# Verificar sem mudar
black --check src/
```

### isort - Organizar imports

```bash
# Organizar imports
isort src/

# Verificar sem mudar
isort --check-only src/
```

### Pylint - Linting

```bash
# Verificar qualidade
pylint src/ --disable=too-few-public-methods

# Gerar relatório
pylint src/ > lint-report.txt
```

### mypy - Type checking

```bash
# Verificar tipos
mypy src/ --ignore-missing-imports

# Strict mode
mypy src/ --strict
```

## 🔄 Git Workflow Desenvolvimento

### 1. Crie branch para feature

```bash
git checkout develop
git pull origin develop
git checkout -b feat/my-feature
```

### 2. Trabalhe e commit

```bash
# Edite arquivos...
git add .
git commit -m "feat(module): add my feature"
```

### 3. Antes do Push

```bash
# Formatar código
black src/
isort src/

# Verificar qualidade
pylint src/
mypy src/

# Rodar testes
pytest tests/ -v

# Commit qualquer mudança de formatação
git add .
git commit -m "style: format code with black/isort"
```

### 4. Push e PR

```bash
git push origin feat/my-feature
# Vá ao GitHub e crie PR para develop
```

## 🔍 Debugging

### VS Code Debugger

Criar `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Run Script",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": true
    },
    {
      "name": "Python: Run Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["${file}", "-v"],
      "console": "integratedTerminal"
    }
  ]
}
```

### IPython REPL

```bash
# Iniciar IPython (melhor que python shell)
ipython

# No código, para no ponto:
from IPython import embed; embed()
```

### Print Debugging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Variable value: %s", my_var)
```

## 📊 Estrutura de Diretórios

```
LibrIA/
├── src/                    # Código principal
│   ├── data_collection/    # Coleta de dados
│   ├── data_processing/    # Processamento
│   ├── model_training/     # Treinamento
│   └── inference/          # Inferência
├── tests/                  # Testes
├── docs/                   # Documentação
├── model/                  # Modelos treinados
├── data/                   # Dados brutos
└── venv/                   # Ambiente virtual (não commitar)
```

## 🚨 Problemas Comuns

### "ModuleNotFoundError: No module named 'src'"

**Solução:**
```bash
# Instale em modo desenvolvimento
pip install -e .

# Ou adicione ao PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/LibrIA"
```

### "pip install" muito lento

**Solução:**
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
