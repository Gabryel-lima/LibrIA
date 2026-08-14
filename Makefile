################################################################################
#                           🧠 LibrIA - Makefile                               #
#                 Reconhecimento e Tradução de Libras                          #
################################################################################
#
# Fluxo principal:   make setup → make collect → make train → make infer
# Menu completo:     make help
#
# Todo alvo do pipeline tem um comando equivalente no main.py com o MESMO nome:
#   make train-temporal  ==  python main.py train-temporal
#
################################################################################

.DEFAULT_GOAL := help

.PHONY: help setup install-cpu install-gpu install-dev verify environment dirs \
	collect collect-static collect-temporal collect-words collect-unknown report \
	train train-static train-temporal infer infer-static infer-temporal all \
	embedded-train embedded-export embedded-check \
	checkerboard checkerboard-show calibrate-capture calibrate \
	test lint format clean clean-all freeze

# ============================================================================
# VARIÁVEIS
# ============================================================================

PYTHON := python3
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
PROJECT_NAME := LibrIA
PROJECT_VERSION := 2.0.0

# --- Coleta ---------------------------------------------------------------
# Quem está sinalizando. Sem isso não dá para dividir treino/validação/teste
# por pessoa (ver docs/FASE1_RECONHECIMENTO.md).
SUBJECT ?= desconhecido
CAMERA_ID ?= desconhecido
ENVIRONMENT ?= desconhecido
DOMINANT_HAND ?= desconhecido
CAMERA ?= 0
SAMPLES ?= 30
SEQUENCES ?= 30

CAPTURE_ARGS = --subject $(SUBJECT) --camera-id $(CAMERA_ID) \
	--environment $(ENVIRONMENT) --dominant-hand $(DOMINANT_HAND) \
	--camera-index $(CAMERA) --samples $(SAMPLES) --sequences $(SEQUENCES)

# --- Calibração de câmera -------------------------------------------------
CALIBRATION_IMAGES ?= calibration/*.jpg
CALIBRATION_COLS ?= 9
CALIBRATION_ROWS ?= 6
CALIBRATION_DIR ?= calibration
CALIBRATION_TARGET ?= 15
CHECKERBOARD ?= output/checkerboard_9x6.png
CHECKERBOARD_SQUARE ?= 80

# --- Cores ----------------------------------------------------------------
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
MAGENTA := \033[0;35m
CYAN := \033[0;36m
BOLD := \033[1m
NC := \033[0m

HAS_CUDA := $(shell nvidia-smi > /dev/null 2>&1 && echo 1 || echo 0)
ifeq ($(HAS_CUDA),1)
    GPU_STATUS := $(GREEN)✓ CUDA detectado$(NC)
    DEFAULT_INSTALL := install-gpu
else
    GPU_STATUS := $(YELLOW)apenas CPU$(NC)
    DEFAULT_INSTALL := install-cpu
endif

# Checagem instantânea: falha cedo e com mensagem clara se o venv não existe.
# Não importa bibliotecas — isso levaria ~10s em todo comando.
define require_venv
@if [ ! -x "$(VENV_PYTHON)" ]; then \
	echo "$(RED)✗ Ambiente virtual não encontrado. Rode: make setup$(NC)"; exit 1; \
fi
endef

################################################################################
# AJUDA
################################################################################

help: ## 📖 Mostra este menu
	@echo ""
	@echo "$(BOLD)$(BLUE)╔═══════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BOLD)$(BLUE)║$(NC)      🧠  $(BOLD)LibrIA$(NC) — Reconhecimento e Tradução de Libras         $(BOLD)$(BLUE)║$(NC)"
	@echo "$(BOLD)$(BLUE)╚═══════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(BOLD)$(GREEN)● PIPELINE — o caminho principal$(NC)"
	@grep -E '^(collect|report|train|infer|all)[a-z-]*:.*?##' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  $(GREEN)%-18s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)$(BLUE)● SETUP$(NC)"
	@grep -E '^(setup|install|verify|environment|dirs)[a-z-]*:.*?##' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  $(BLUE)%-18s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)$(MAGENTA)● EMBEDDED — TFLite INT8 e Raspberry Pi Pico$(NC)"
	@grep -E '^embedded-[a-z]*:.*?##' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  $(MAGENTA)%-18s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)$(YELLOW)● CÂMERA — calibração$(NC)"
	@grep -E '^(checkerboard|calibrate)[a-z-]*:.*?##' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  $(YELLOW)%-18s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)$(CYAN)● DESENVOLVIMENTO$(NC)"
	@grep -E '^(test|lint|format|clean|freeze)[a-z-]*:.*?##' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  $(CYAN)%-18s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Coleta — sempre identifique a pessoa:$(NC)"
	@echo "  $(CYAN)make collect SUBJECT=ana ENVIRONMENT=sala CAMERA_ID=c920 DOMINANT_HAND=right$(NC)"
	@echo ""
	@echo "$(BOLD)Variáveis:$(NC) SUBJECT CAMERA_ID ENVIRONMENT DOMINANT_HAND CAMERA SAMPLES SEQUENCES"
	@echo "$(BOLD)GPU:$(NC) $(GPU_STATUS)"
	@echo ""

################################################################################
# SETUP
################################################################################

setup: $(VENV_DIR) ## 🔧 Cria o venv e instala as dependências
	@echo "$(BLUE)→ Instalando dependências ($(DEFAULT_INSTALL))...$(NC)"
	@$(MAKE) --no-print-directory $(DEFAULT_INSTALL)
	@$(MAKE) --no-print-directory dirs
	@echo "$(GREEN)✓ Setup completo. Rode 'make verify' para validar.$(NC)"

$(VENV_DIR):
	@echo "$(BLUE)→ Criando ambiente virtual...$(NC)"
	@$(PYTHON) -m venv $(VENV_DIR)
	@$(VENV_PIP) install --upgrade pip setuptools wheel --quiet

install-cpu: ## 💾 Instala dependências (CPU)
	@$(VENV_PIP) install -r requirements.txt && echo "$(GREEN)✓ Instalação CPU completa$(NC)"

install-gpu: ## 🚀 Instala dependências com CUDA
	@$(VENV_PIP) install -r requirements.txt -r requirements-gpu.txt && echo "$(GREEN)✓ Instalação GPU completa$(NC)"

install-dev: ## 📚 Instala dependências de desenvolvimento
	@$(VENV_PIP) install -r requirements-dev.txt --quiet && echo "$(GREEN)✓ Dependências dev instaladas$(NC)"

verify: ## ✓ Verifica se o ambiente está utilizável
	$(require_venv)
	@echo "$(BLUE)→ Verificando ambiente...$(NC)"
	@echo "  Python:       $(shell $(VENV_PYTHON) --version 2>&1)"
	@$(VENV_PYTHON) -c "import cv2; print('  OpenCV:      ', cv2.__version__)" 2>/dev/null || echo "  $(RED)OpenCV: ausente$(NC)"
	@$(VENV_PYTHON) -c "import sklearn; print('  Scikit-learn:', sklearn.__version__)" 2>/dev/null || echo "  $(RED)Scikit-learn: ausente$(NC)"
	@$(VENV_PYTHON) -c "import numpy; print('  NumPy:       ', numpy.__version__)" 2>/dev/null || echo "  $(RED)NumPy: ausente$(NC)"
	@$(VENV_PYTHON) -c "import tensorflow; print('  TensorFlow:  ', tensorflow.__version__)" 2>/dev/null || echo "  $(YELLOW)TensorFlow: ausente (necessário para os modelos temporais)$(NC)"
	@$(VENV_PYTHON) -c "import mediapipe; print('  MediaPipe:   ', mediapipe.__version__)" 2>/dev/null || echo "  $(YELLOW)MediaPipe: ausente (necessário para coleta e inferência; requer AVX)$(NC)"
	@$(VENV_PYTHON) -c "from config.settings import validate_config; e = validate_config(); print('  Config:       ok' if not e else '  Config: ' + str(e))"
	@echo "$(GREEN)✓ Verificação concluída$(NC)"

environment: ## 📊 Exibe informações do ambiente
	@echo "$(BOLD)$(CYAN)  $(PROJECT_NAME) v$(PROJECT_VERSION)$(NC)"
	@echo "  Python:   $(shell $(VENV_PYTHON) --version 2>/dev/null)"
	@echo "  Sistema:  $(shell uname -s) $(shell uname -m)"
	@echo "  GPU:      $(GPU_STATUS)"
	@echo "  Dataset:  ./dataset  |  Modelos: ./model  |  Saídas: ./output"

dirs: ## 📂 Cria a estrutura de diretórios
	@mkdir -p dataset/static dataset/temporal model output training_plots
	@echo "$(GREEN)✓ Diretórios prontos$(NC)"

################################################################################
# PIPELINE — COLETA
################################################################################

collect: dirs ## 📷 Coleta o dataset mínimo (alfabeto estático + J/Z)
	$(require_venv)
	@echo "$(YELLOW)→ Coleta do dataset mínimo | pessoa: $(SUBJECT) | ambiente: $(ENVIRONMENT)$(NC)"
	@$(VENV_PYTHON) main.py collect $(CAPTURE_ARGS)

collect-static: dirs ## 📷 Coleta o alfabeto manual estático (24 letras)
	$(require_venv)
	@echo "$(YELLOW)→ Coleta estática | pessoa: $(SUBJECT) | $(SAMPLES) amostras/classe$(NC)"
	@$(VENV_PYTHON) main.py collect-static $(CAPTURE_ARGS)

collect-temporal: dirs ## 📷 Coleta as letras temporais (J e Z)
	$(require_venv)
	@echo "$(YELLOW)→ Coleta temporal | pessoa: $(SUBJECT) | $(SEQUENCES) sequências/classe$(NC)"
	@$(VENV_PYTHON) main.py collect-temporal $(CAPTURE_ARGS)

collect-words: dirs ## 📷 Coleta palavras e gestos funcionais (vocabulário lexical)
	$(require_venv)
	@echo "$(YELLOW)→ Coleta de palavras | pessoa: $(SUBJECT)$(NC)"
	@$(VENV_PYTHON) main.py collect-words $(CAPTURE_ARGS)

collect-unknown: dirs ## 📷 Coleta amostras fora do vocabulário (classe de rejeição)
	$(require_venv)
	@echo "$(YELLOW)→ Coleta de negativos | pessoa: $(SUBJECT)$(NC)"
	@$(VENV_PYTHON) main.py collect-unknown $(CAPTURE_ARGS)

report: dirs ## 📊 Cobertura do dataset: vocabulário, metadados e divisão por pessoa
	$(require_venv)
	@$(VENV_PYTHON) -m scripts.dataset_report --json output/dataset_report.json

################################################################################
# PIPELINE — TREINO
################################################################################

train: dirs ## 🧠 Treina os modelos estático e temporal
	$(require_venv)
	@$(VENV_PYTHON) main.py train

train-static: dirs ## 🤖 Treina só o Random Forest estático
	$(require_venv)
	@$(VENV_PYTHON) main.py train-static

train-temporal: dirs ## 🧠 Treina só a LSTM temporal
	$(require_venv)
	@$(VENV_PYTHON) main.py train-temporal

################################################################################
# PIPELINE — INFERÊNCIA
################################################################################

infer: ## 🎯 Inferência híbrida em tempo real (recomendado) — 'q' para sair
	$(require_venv)
	@$(VENV_PYTHON) main.py infer --camera-index $(CAMERA)

infer-static: ## 🎯 Inferência só com o modelo estático
	$(require_venv)
	@$(VENV_PYTHON) main.py infer-static --camera-index $(CAMERA)

infer-temporal: ## 🎯 Inferência só com o modelo temporal
	$(require_venv)
	@$(VENV_PYTHON) main.py infer-temporal --camera-index $(CAMERA)

all: dirs ## ▶️  Pipeline completo: coletar → treinar → inferir
	$(require_venv)
	@$(VENV_PYTHON) main.py all $(CAPTURE_ARGS)

################################################################################
# EMBEDDED
################################################################################

embedded-train: dirs ## 📦 Treina as CNNs quantizadas e exporta o bundle
	$(require_venv)
	@$(VENV_PYTHON) main.py embedded-train

embedded-export: dirs ## 📦 Empacota modelos, manifesto e pacote C/C++ do Pico
	$(require_venv)
	@$(VENV_PYTHON) main.py embedded-export

embedded-check: ## 🔎 Valida o bundle embedded sobre os datasets NPY
	$(require_venv)
	@$(VENV_PYTHON) main.py embedded-check

################################################################################
# CÂMERA — CALIBRAÇÃO
################################################################################

checkerboard: dirs ## 🧾 Gera a imagem do tabuleiro de calibração
	$(require_venv)
	@$(VENV_PYTHON) -m scripts.generate_checkerboard --cols $(CALIBRATION_COLS) --rows $(CALIBRATION_ROWS) --square-size $(CHECKERBOARD_SQUARE) --output $(CHECKERBOARD)

checkerboard-show: ## 🖥️  Exibe o tabuleiro em tela cheia
	$(require_venv)
	@$(VENV_PYTHON) -m scripts.show_checkerboard --image $(CHECKERBOARD)

calibrate-capture: ## 📷 Captura fotos do tabuleiro pela webcam e calibra
	$(require_venv)
	@$(VENV_PYTHON) -m scripts.calibrate_camera --capture --capture-dir $(CALIBRATION_DIR) --target-images $(CALIBRATION_TARGET) --camera-index $(CAMERA) --cols $(CALIBRATION_COLS) --rows $(CALIBRATION_ROWS)

calibrate: ## 🎥 Calibra a câmera a partir de imagens já capturadas
	$(require_venv)
	@$(VENV_PYTHON) -m scripts.calibrate_camera $(CALIBRATION_IMAGES) --cols $(CALIBRATION_COLS) --rows $(CALIBRATION_ROWS)

################################################################################
# DESENVOLVIMENTO
################################################################################

test: ## ✓ Roda a suíte de testes
	$(require_venv)
	@$(VENV_PYTHON) -m unittest discover -s tests -t . -v

lint: ## 🔍 flake8 (requer make install-dev)
	@if [ ! -x "$(VENV_DIR)/bin/flake8" ]; then echo "$(RED)✗ flake8 ausente. Rode: make install-dev$(NC)"; exit 1; fi
	@$(VENV_DIR)/bin/flake8 src/ scripts/ config/ main.py --max-line-length=119 --exclude=__pycache__

format: ## 🎨 black (requer make install-dev)
	@if [ ! -x "$(VENV_DIR)/bin/black" ]; then echo "$(RED)✗ black ausente. Rode: make install-dev$(NC)"; exit 1; fi
	@$(VENV_DIR)/bin/black src/ scripts/ config/ main.py

freeze: ## 📌 Gera requirements.lock com as versões exatas
	@$(VENV_PIP) freeze > requirements.lock
	@echo "$(GREEN)✓ requirements.lock gerado$(NC)"

################################################################################
# LIMPEZA
################################################################################

clean: ## 🧹 Remove caches e saídas temporárias
	@find . -path ./$(VENV_DIR) -prune -o -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	@find . -path ./$(VENV_DIR) -prune -o -type f -name "*.py[co]" -delete 2>/dev/null; true
	@rm -rf .pytest_cache .mypy_cache
	@rm -f output/*.mp4 output/*.jpg
	@echo "$(GREEN)✓ Limpeza concluída$(NC)"

clean-all: clean ## 💣 Remove venv, dataset e modelos (CUIDADO!)
	@echo "$(RED)⚠️  Isto remove: $(VENV_DIR)/, dataset/, model/, output/, training_plots/$(NC)"
	@read -p "Digite 'confirmar' para continuar: " confirm; \
	if [ "$$confirm" = "confirmar" ]; then \
		rm -rf $(VENV_DIR) dataset model output training_plots requirements.lock; \
		echo "$(GREEN)✓ Removido$(NC)"; \
	else \
		echo "$(YELLOW)✗ Cancelado$(NC)"; \
	fi

################################################################################
# FIM
################################################################################
