################################################################################
#                           🧠 LibrIA - Makefile                              #
#                 Reconhecimento de Libras com Visão Computacional            #
################################################################################
#
# Este Makefile automatiza o setup, treinamento e inferência do projeto.
# Use 'make help' para ver todos os comandos disponíveis.
#
################################################################################

# ============================================================================
# CONFIGURAÇÕES INICIAIS
# ============================================================================



.PHONY: help setup install install-gpu install-cpu collect-static collect-temporal collect-minimal-dataset generate-checkerboard show-checkerboard capture-calibration calibrate-camera train train-lstm train-embedded train-embedded-temporal train-embedded-all export-embedded train-hybrid infer infer-lstm infer-hybrid infer-embedded run-lstm \
	run test clean clean-all dirs verify-setup environment lint format install-dev freeze update status

# Variáveis de Configuração
PYTHON := python3
PYTHON_VERSION := 3.11
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
PROJECT_NAME := LibrIA
PROJECT_VERSION := 1.0.0
STATIC_DATASET_SIZE ?= 150
STATIC_LABELS ?= A B C D E F G H I K L M N O P Q R S T U V W X Y
STATIC_SAMPLE_COUNT ?= 30
SEQUENCE_LABELS ?= J Z
SEQUENCE_COUNT ?= 30
SEQUENCE_LENGTH ?= 30
SEQUENCE_CAMERA ?= 0
CALIBRATION_IMAGES ?= calibration/*.jpg
CALIBRATION_COLS ?= 9
CALIBRATION_ROWS ?= 6
CALIBRATION_CAMERA ?= 0
CALIBRATION_CAPTURE_DIR ?= calibration
CALIBRATION_TARGET_IMAGES ?= 15
CHECKERBOARD_OUTPUT ?= output/checkerboard_9x6.png
CHECKERBOARD_SQUARE_SIZE ?= 80

# Cores para output (ANSI)
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
MAGENTA := \033[0;35m
CYAN := \033[0;36m
BOLD := \033[1m
NC := \033[0m # No Color

# Detectar CUDA
HAS_CUDA := $(shell nvidia-smi > /dev/null 2>&1 && echo 1 || echo 0)

ifeq ($(HAS_CUDA),1)
    GPU_STATUS := $(GREEN)✓ CUDA Detectado$(NC)
    DEFAULT_INSTALL := install-gpu
else
    GPU_STATUS := $(RED)✗ Apenas CPU$(NC)
    DEFAULT_INSTALL := install-cpu
endif

# ============================================================================
# 📋 LEGENDA DE CORES
# ============================================================================
# Este Makefile usa cores para categorizar os comandos:
#
# $(BLUE)● SETUP$(NC)        - Configuração inicial do ambiente
# $(GREEN)● EXECUTE$(NC)      - Execução do pipeline
# $(YELLOW)● DATA$(NC)        - Processamento e coleta de dados
# $(MAGENTA)● DEVELOPMENT$(NC) - Ferramentas de desenvolvimento
# $(CYAN)● UTILITY$(NC)      - Utilitários e limpeza
#
# ============================================================================

################################################################################
# SETUP & ENVIRONMENT
################################################################################

help: ## 📖 Mostra este menu de ajuda com todos os comandos
	@echo ""
	@echo "$(BOLD)$(BLUE)╔═══════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BOLD)$(BLUE)║$(NC)           $(BOLD)🧠  LibrIA - Makefile de Comandos$(NC)             $(BOLD)$(BLUE)║$(NC)"
	@echo "$(BOLD)$(BLUE)║$(NC)    Reconhecimento de Libras com Visão Computacional      $(BOLD)$(BLUE)║$(NC)"
	@echo "$(BOLD)$(BLUE)╚═══════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(BOLD)$(BLUE)● SETUP - Configuração Inicial:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^setup|^install|^verify/ && !/^$$/ {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(BOLD)$(GREEN)● EXECUTE - Execução do Pipeline:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^run|^collect-static|^collect-temporal|^collect-minimal-dataset|^generate-checkerboard|^show-checkerboard|^capture-calibration|^train|^train-lstm|^train-embedded|^train-embedded-temporal|^train-embedded-all|^export-embedded|^train-hybrid|^infer|^infer-lstm|^infer-hybrid|^infer-embedded|^calibrate-camera/ && !/^setup/ && !/^$$/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(BOLD)$(YELLOW)● TESTING & DEBUG:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^test|^environment|^lint|^format/ && !/^$$/ {printf "  $(MAGENTA)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(BOLD)$(CYAN)● UTILITY - Limpeza & Utilitários:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^clean|^dirs/ && !/^$$/ {printf "  $(CYAN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(BOLD)Status do Sistema:$(NC)"
	@echo "  Python:     $(BLUE)$(PYTHON_VERSION)$(NC)"
	@echo "  GPU:        $(GPU_STATUS)"
	@echo "  venv:       $(BLUE)$(VENV_DIR)$(NC)"
	@echo ""
	@echo "$(BOLD)Exemplo de uso:$(NC)"
	@echo "  $(CYAN)make setup $(NC)                  # Setup inicial"
	@echo "  $(CYAN)make collect-static STATIC_SAMPLE_COUNT=30 $(NC)# Coletar 30 amostras estáticas por classe"
	@echo "  $(CYAN)make collect-temporal $(NC)       # Coletar sequências temporais padrão (J e Z)"
	@echo "  $(CYAN)make collect-minimal-dataset $(NC)# Coletar dataset mínimo completo em dataset/"
	@echo "  $(CYAN)make collect-temporal SEQUENCE_LABELS=J\ Z SEQUENCE_COUNT=20 $(NC)# Coletar 20 sequências de J e Z"
	@echo "  $(CYAN)make generate-checkerboard $(NC)  # Gerar imagem do tabuleiro 9x6"
	@echo "  $(CYAN)make show-checkerboard $(NC) 	# Exibir o tabuleiro em tela cheia"
	@echo "  $(CYAN)make capture-calibration $(NC)    # Abrir webcam, salvar fotos do tabuleiro e calibrar"
	@echo "  $(CYAN)make calibrate-camera CALIBRATION_IMAGES='calibration/*.jpg' $(NC)# Calibrar usando imagens do tabuleiro"
	@echo "  $(CYAN)make train $(NC)           	# Treinar modelo Random Forest com dataset estático"
	@echo "  $(CYAN)make train-lstm $(NC)             # Treinar modelo temporal LSTM"
	@echo "  $(CYAN)make train-embedded $(NC) 	# Treinar CNN estática embedded via sample_XXX.npy"
	@echo "  $(CYAN)make train-embedded-temporal $(NC)# Treinar CNN temporal embedded para J e Z"
	@echo "  $(CYAN)make train-embedded-all $(NC) 	# Treinar os dois modelos embedded em sequência"
	@echo "  $(CYAN)make export-embedded $(NC)   	# Empacotar os dois modelos quantizados e metadados"
	@echo "  $(CYAN)make train-hybrid $(NC)      	# Retreinar modelos estático + temporal"
	@echo "  $(CYAN)make run-lstm $(NC)         	# Pipeline temporal completo"
	@echo "  $(CYAN)make run $(NC)            	# Executar pipeline completo"
	@echo "  $(CYAN)make infer $(NC)          	# Inferência em tempo real"
	@echo "  $(CYAN)make infer-lstm $(NC)     	# Inferência temporal em tempo real"
	@echo "  $(CYAN)make infer-hybrid $(NC)   	# Inferência híbrida com arbitragem"
	@echo "  $(CYAN)make infer-embedded $(NC) 	# Verificar o bundle embedded em cima dos datasets NPY"
	@echo ""

setup: $(VENV_DIR) ## 🔧 Setup inicial - cria venv e instala dependências
	@echo "$(GREEN)✓ Ambiente virtual criado em $(VENV_DIR)$(NC)"
	@echo "$(BLUE)→ Instalando dependências ($(DEFAULT_INSTALL))...$(NC)"
	@$(MAKE) --no-print-directory $(DEFAULT_INSTALL)
	@echo "$(GREEN)✓ Setup completo! Use 'make verify-setup' para validar.$(NC)"
	@echo "$(YELLOW)→ Não esqueça de ativar o .venv$(NC)"

$(VENV_DIR): ## Cria o ambiente virtual (dependência interna)
	@echo "$(BLUE)→ Criando ambiente virtual com Python $(PYTHON_VERSION)...$(NC)"
	@$(PYTHON) -m venv $(VENV_DIR)
	@$(VENV_PIP) install --upgrade pip setuptools wheel --quiet
	@echo "$(GREEN)✓ Ambiente virtual criado$(NC)"

install-cpu: ## 💾 Instala dependências para CPU
	@echo "$(YELLOW)→ Instalando LibrIA com suporte CPU...$(NC)"
	@$(VENV_PIP) install -r requirements.txt && \
	echo "$(GREEN)✓ Instalação CPU completa$(NC)" || \
	(echo "$(RED)✗ Erro na instalação$(NC)" && exit 1)

install-gpu: ## 🚀 Instala dependências com suporte GPU (CUDA 12.4)
	@echo "$(YELLOW)→ Instalando LibrIA com suporte GPU (CUDA 12.4)...$(NC)"
	@$(VENV_PIP) install -r requirements.txt -r requirements-gpu.txt && \
	echo "$(GREEN)✓ Instalação GPU completa$(NC)" || \
	(echo "$(RED)✗ Erro na instalação GPU$(NC)" && exit 1)

verify-setup: ## ✓ Verifica se o ambiente está configurado corretamente
	@echo "$(BLUE)→ Verificando setup do projeto...$(NC)"
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "$(RED)✗ Ambiente virtual não encontrado!$(NC)"; exit 1; \
	fi
	@if ! $(VENV_PYTHON) -c "import torch, sklearn, cv2" 2>/dev/null; then \
		echo "$(RED)✗ Alguma dependência está faltando!$(NC)"; exit 1; \
	fi
	@echo "$(GREEN)✓ Python: $(shell $(VENV_PYTHON) --version)$(NC)"
	@if $(VENV_PYTHON) -c "import tensorflow" 2>/dev/null; then \
		echo "$(GREEN)✓ TensorFlow: $(shell $(VENV_PYTHON) -c 'import tensorflow; print(tensorflow.__version__)' 2>/dev/null)$(NC)"; \
	else \
		echo "$(YELLOW)⚠ TensorFlow: Não disponível (pode ser incompatível com esta CPU)$(NC)"; \
	fi
	@echo "$(GREEN)✓ PyTorch: $(shell $(VENV_PYTHON) -c 'import torch; print(torch.__version__)' 2>/dev/null)$(NC)"
	@echo "$(GREEN)✓ OpenCV: $(shell $(VENV_PYTHON) -c 'import cv2; print(cv2.__version__)' 2>/dev/null)$(NC)"
	@echo "$(GREEN)✓ Scikit-learn: $(shell $(VENV_PYTHON) -c 'import sklearn; print(sklearn.__version__)' 2>/dev/null)$(NC)"
	@if $(VENV_PYTHON) -c "import mediapipe" 2>/dev/null; then \
		echo "$(GREEN)✓ MediaPipe: $(shell $(VENV_PYTHON) -c 'import mediapipe; print(mediapipe.__version__)' 2>/dev/null)$(NC)"; \
	else \
		echo "$(YELLOW)⚠ MediaPipe: Não disponível (requer suporte AVX)$(NC)"; \
	fi
	@if $(VENV_PYTHON) -c "import keras" 2>/dev/null; then \
		echo "$(GREEN)✓ Keras: $(shell $(VENV_PYTHON) -c 'import keras; print(keras.__version__)' 2>/dev/null)$(NC)"; \
	else \
		echo "$(YELLOW)⚠ Keras: Não disponível (dependência do TensorFlow)$(NC)"; \
	fi
	@$(VENV_PYTHON) test_setup.py && echo "$(GREEN)✓ Testes de setup passaram!$(NC)" || echo "$(RED)✗ Testes falharam$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)✓ Ambiente pronto para uso! Execute 'make help' para mais info.$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"

environment: ## 📊 Exibe informações do ambiente
	@echo "$(BOLD)$(CYAN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(BOLD)$(CYAN)  Informações do Ambiente LibrIA$(NC)"
	@echo "$(BOLD)$(CYAN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(BLUE)Project:$(NC)"
	@echo "  Nome:               $(BLUE)$(PROJECT_NAME)$(NC)"
	@echo "  Versão:             $(BLUE)$(PROJECT_VERSION)$(NC)"
	@echo ""
	@echo "$(GREEN)Python:$(NC)"
	@echo "  Executável:         $(BLUE)$(VENV_PYTHON)$(NC)"
	@echo "  Versão:             $(BLUE)$(shell $(VENV_PYTHON) --version 2>/dev/null)$(NC)"
	@echo "  pip:                $(BLUE)$(shell $(VENV_PIP) --version 2>/dev/null)$(NC)"
	@echo ""
	@echo "$(YELLOW)Sistema:$(NC)"
	@echo "  OS:                 $(BLUE)$(shell uname -s)$(NC)"
	@echo "  Arquitetura:        $(BLUE)$(shell uname -m)$(NC)"
	@echo "  GPU:                $(GPU_STATUS)"
	@if [ "$(HAS_CUDA)" = "1" ]; then \
		echo "  CUDA Version:       $(BLUE)$(shell nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)$(NC)"; \
	fi
	@echo ""
	@echo "$(MAGENTA)Diretórios:$(NC)"
	@echo "  Venv:               $(BLUE)$(VENV_DIR)$(NC)"
	@echo "  Dados:              $(BLUE)./data$(NC)"
	@echo "  Dataset:            $(BLUE)./dataset$(NC)"
	@echo "  Modelos:            $(BLUE)./model$(NC)"
	@echo "  Output:             $(BLUE)./output$(NC)"
	@echo ""

################################################################################
# PIPELINE - COLLECT, TRAIN, INFER
################################################################################

dirs: ## 📂 Cria a estrutura de diretórios do projeto
	@echo "$(BLUE)→ Criando estrutura de diretórios...$(NC)"
	@mkdir -p dataset dataset/static dataset/temporal model output training_plots
	@echo "$(GREEN)✓ Diretórios criados: dataset/static, dataset/temporal, model/, output/, training_plots/$(NC)"

collect-static: verify-setup dirs ## 📷 Coleta dataset estático unificado em dataset/static
	@echo "$(YELLOW)→ Coletando dataset estático unificado...$(NC)"
	@echo "$(CYAN)  Labels: $(STATIC_LABELS) | Amostras por classe: $(STATIC_SAMPLE_COUNT) | Câmera: $(SEQUENCE_CAMERA)$(NC)"
	@$(VENV_PYTHON) -m scripts.collect_dataset static --labels $(STATIC_LABELS) --samples-per-label $(STATIC_SAMPLE_COUNT) --camera-index $(SEQUENCE_CAMERA)

collect-temporal: verify-setup dirs ## 📷 Coleta dataset temporal unificado em dataset/temporal
	@echo "$(YELLOW)→ Coletando dataset temporal unificado...$(NC)"
	@echo "$(CYAN)  Labels: $(SEQUENCE_LABELS) | Sequências: $(SEQUENCE_COUNT) | Frames válidos: $(SEQUENCE_LENGTH) | Câmera: $(SEQUENCE_CAMERA)$(NC)"
	@$(VENV_PYTHON) -m scripts.collect_dataset temporal --labels $(SEQUENCE_LABELS) --num-sequences $(SEQUENCE_COUNT) --seq-length $(SEQUENCE_LENGTH) --camera-index $(SEQUENCE_CAMERA)

collect-minimal-dataset: verify-setup dirs ## 📷 Coleta o dataset mínimo completo em dataset/
	@echo "$(YELLOW)→ Coletando dataset mínimo completo...$(NC)"
	@$(MAKE) --no-print-directory collect-static STATIC_SAMPLE_COUNT=$(STATIC_SAMPLE_COUNT) SEQUENCE_CAMERA=$(SEQUENCE_CAMERA)
	@$(MAKE) --no-print-directory collect-temporal SEQUENCE_COUNT=$(SEQUENCE_COUNT) SEQUENCE_LENGTH=$(SEQUENCE_LENGTH) SEQUENCE_CAMERA=$(SEQUENCE_CAMERA)

generate-checkerboard: verify-setup dirs ## 🧾 Gera uma imagem de tabuleiro para calibração de câmera
	@echo "$(YELLOW)→ Gerando tabuleiro de calibração...$(NC)"
	@echo "$(CYAN)  Saída: $(CHECKERBOARD_OUTPUT) | Padrão: $(CALIBRATION_COLS)x$(CALIBRATION_ROWS) | Quadrado: $(CHECKERBOARD_SQUARE_SIZE)px$(NC)"
	@$(VENV_PYTHON) -m scripts.generate_checkerboard --cols $(CALIBRATION_COLS) --rows $(CALIBRATION_ROWS) --square-size $(CHECKERBOARD_SQUARE_SIZE) --output $(CHECKERBOARD_OUTPUT)

show-checkerboard: verify-setup dirs ## 🖥️ Exibe o tabuleiro gerado em tela cheia
	@echo "$(YELLOW)→ Exibindo tabuleiro em tela cheia...$(NC)"
	@echo "$(CYAN)  Imagem: $(CHECKERBOARD_OUTPUT)$(NC)"
	@$(VENV_PYTHON) -m scripts.show_checkerboard --image $(CHECKERBOARD_OUTPUT)

capture-calibration: verify-setup dirs ## 📷 Abre a webcam, captura imagens do tabuleiro e calibra a câmera
	@echo "$(YELLOW)→ Capturando imagens para calibração...$(NC)"
	@echo "$(CYAN)  Câmera: $(CALIBRATION_CAMERA) | Saída: $(CALIBRATION_CAPTURE_DIR) | Alvo: $(CALIBRATION_TARGET_IMAGES) imagens$(NC)"
	@$(VENV_PYTHON) -m scripts.calibrate_camera --capture --capture-dir $(CALIBRATION_CAPTURE_DIR) --target-images $(CALIBRATION_TARGET_IMAGES) --camera-index $(CALIBRATION_CAMERA) --cols $(CALIBRATION_COLS) --rows $(CALIBRATION_ROWS)

calibrate-camera: verify-setup dirs ## 🎥 Calibra a câmera por imagens do tabuleiro
	@echo "$(YELLOW)→ Iniciando calibração da câmera...$(NC)"
	@echo "$(MAGENTA)  Imagens: $(CALIBRATION_IMAGES) | Padrão do tabuleiro: $(CALIBRATION_COLS)x$(CALIBRATION_ROWS)$(NC)"
	@$(VENV_PYTHON) -m scripts.calibrate_camera $(CALIBRATION_IMAGES) --cols $(CALIBRATION_COLS) --rows $(CALIBRATION_ROWS)

train: verify-setup ## 🤖 Treina modelo Random Forest
	@echo "$(MAGENTA)→ Iniciando treinamento do modelo...$(NC)"
	@$(VENV_PYTHON) main.py train
	@echo "$(GREEN)✓ Modelo treinado! Salvo em ./model/model.pickle$(NC)"

train-lstm: verify-setup dirs ## 🧠 Treina modelo temporal LSTM com dataset/temporal
	@echo "$(MAGENTA)→ Iniciando treinamento do modelo temporal...$(NC)"
	@$(VENV_PYTHON) main.py train_lstm
	@echo "$(GREEN)✓ Modelo temporal treinado! Salvo em ./model/libras_lstm.keras$(NC)"

train-embedded: verify-setup dirs ## 📦 Treina CNN estática quantizada para deployment embedded
	@echo "$(MAGENTA)→ Iniciando treinamento embedded...$(NC)"
	@$(VENV_PYTHON) main.py train_embedded
	@echo "$(GREEN)✓ Modelo embedded exportado em ./model/libria_embedded_cnn_int8.tflite$(NC)"

train-embedded-temporal: verify-setup dirs ## 🎞️ Treina CNN temporal quantizada para J e Z em embedded
	@echo "$(MAGENTA)→ Iniciando treinamento embedded temporal...$(NC)"
	@$(VENV_PYTHON) main.py train_embedded_temporal
	@echo "$(GREEN)✓ Modelo temporal embedded exportado em ./model/libria_embedded_temporal_cnn_int8.tflite$(NC)"

train-embedded-all: verify-setup dirs ## 🧩 Treina os modelos embedded estático e temporal em sequência
	@echo "$(MAGENTA)→ Iniciando treinamento embedded completo...$(NC)"
	@$(VENV_PYTHON) main.py train_embedded_all
	@echo "$(GREEN)✓ Modelos embedded estático e temporal atualizados!$(NC)"

export-embedded: verify-setup dirs ## 📦 Empacota modelos, metadados e pacote C/C++ pronto para o Pico
	@echo "$(MAGENTA)→ Exportando bundle embedded...$(NC)"
	@$(VENV_PYTHON) main.py export_embedded
	@echo "$(GREEN)✓ Bundle embedded e pacote Pico exportados em ./model/embedded_bundle$(NC)"

train-hybrid: verify-setup dirs ## 🧠♻️ Retreina os modelos estático e temporal
	@echo "$(MAGENTA)→ Iniciando retreinamento híbrido...$(NC)"
	@$(VENV_PYTHON) main.py train_hybrid
	@echo "$(GREEN)✓ Modelos híbridos atualizados!$(NC)"

infer: verify-setup ## 🎯 Inferência em tempo real com webcam
	@echo "$(GREEN)→ Iniciando inferência em tempo real...$(NC)"
	@echo "$(MAGENTA)  Pressione 'q' para sair$(NC)"
	@$(VENV_PYTHON) main.py infer

infer-lstm: verify-setup ## 🎯 Inferência temporal LSTM em tempo real
	@echo "$(GREEN)→ Iniciando inferência temporal em tempo real...$(NC)"
	@echo "$(MAGENTA)  Pressione 'q' para sair$(NC)"
	@$(VENV_PYTHON) main.py infer_lstm

infer-hybrid: verify-setup ## 🎯 Inferência híbrida em tempo real
	@echo "$(GREEN)→ Iniciando inferência híbrida em tempo real...$(NC)"
	@echo "$(MAGENTA)  Pressione 'q' para sair$(NC)"
	@$(VENV_PYTHON) main.py infer_hybrid

infer-embedded: verify-setup ## 🎯 Verifica o bundle embedded usando os datasets NPY
	@echo "$(GREEN)→ Verificando runtime embedded...$(NC)"
	@$(VENV_PYTHON) main.py infer_embedded

run-lstm: verify-setup dirs ## ▶️  Pipeline temporal: coletar, treinar, inferir
	@echo "$(BOLD)$(GREEN)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BOLD)$(GREEN)║$(NC)         Executando Pipeline Temporal de LibrIA       $(BOLD)$(GREEN)║$(NC)"
	@echo "$(BOLD)$(GREEN)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@$(MAKE) --no-print-directory collect-temporal
	@$(MAKE) --no-print-directory train-lstm
	@$(MAKE) --no-print-directory infer-lstm

run: verify-setup dirs ## ▶️  Executa pipeline completo (collect-minimal→train-hybrid→infer-hybrid)
	@echo "$(BOLD)$(GREEN)╔════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BOLD)$(GREEN)║$(NC)       Executando Pipeline Completo de LibrIA          $(BOLD)$(GREEN)║$(NC)"
	@echo "$(BOLD)$(GREEN)╚════════════════════════════════════════════════════════════════╝$(NC)"
	@$(VENV_PYTHON) main.py all
	@echo "$(BOLD)$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)✓ Pipeline completo finalizado com sucesso!$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"

################################################################################
# TESTING & QUALITY
################################################################################

test: verify-setup ## ✓ Executa testes de setup
	@echo "$(MAGENTA)→ Executando testes...$(NC)"
	@$(VENV_PYTHON) test_setup.py
	@echo "$(GREEN)✓ Testes completados$(NC)"

lint: ## 🔍 Verifica código com flake8 (requer requirements-dev.txt)
	@if [ ! -f "$(VENV_DIR)/bin/flake8" ]; then \
		echo "$(RED)✗ flake8 não instalado. Execute: make install-dev$(NC)"; exit 1; \
	fi
	@echo "$(MAGENTA)→ Verificando qualidade do código...$(NC)"
	@$(VENV_DIR)/bin/flake8 src/ main.py --max-line-length=119 --exclude=__pycache__

format: ## 🎨 Formata código com black (requer requirements-dev.txt)
	@if [ ! -f "$(VENV_DIR)/bin/black" ]; then \
		echo "$(RED)✗ black não instalado. Execute: make install-dev$(NC)"; exit 1; \
	fi
	@echo "$(MAGENTA)→ Formatando código...$(NC)"
	@$(VENV_DIR)/bin/black src/ main.py

install-dev: ## 📚 Instala dependências de desenvolvimento
	@echo "$(YELLOW)→ Instalando dependências de desenvolvimento...$(NC)"
	@$(VENV_PIP) install -r requirements-dev.txt --quiet
	@echo "$(GREEN)✓ Dependências dev instaladas$(NC)"

################################################################################
# CLEANUP
################################################################################

clean: ## 🧹 Remove arquivos temporários e __pycache__
	@echo "$(CYAN)→ Limpando arquivos temporários...$(NC)"
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null; true
	@find . -type f -name ".coverage" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; true
	@rm -f output/*.mp4 output/*.jpg 2>/dev/null; true
	@echo "$(GREEN)✓ Limpeza concluída$(NC)"

clean-all: clean ## 💣 Remove venv, dados e modelos (CUIDADO!)
	@echo "$(RED)⚠️  AVISO: Você está prestes a remover:$(NC)"
	@echo "  - Ambiente virtual ($(VENV_DIR))"
	@echo "  - Dados coletados (./data)"
	@echo "  - Dataset processado (./dataset)"
	@echo "  - Modelos treinados (./model)"
	@echo ""
	@read -p "Digite 'confirmar' para continuar: " confirm; \
	if [ "$$confirm" = "confirmar" ]; then \
		echo "$(RED)→ Removendo...$(NC)"; \
		rm -rf $(VENV_DIR); \
		rm -rf data dataset model output training_plots; \
		rm -f requirements.lock; \
		echo "$(GREEN)✓ Limpeza completa realizada$(NC)"; \
	else \
		echo "$(YELLOW)✗ Operação cancelada$(NC)"; \
	fi

################################################################################
# ADDITIONAL COMMANDS
################################################################################

freeze: ## 📌 Gera lock file com versões exatas (requirements.lock)
	@echo "$(BLUE)→ Gerando requirements.lock...$(NC)"
	@$(VENV_PIP) freeze > requirements.lock
	@echo "$(GREEN)✓ Lock file criado: requirements.lock$(NC)"

update: ## 🔄 Atualiza dependências do requirements.txt
	@echo "$(YELLOW)→ Atualizando dependências...$(NC)"
	@$(VENV_PIP) install --upgrade -r requirements.txt --quiet
	@echo "$(GREEN)✓ Dependências atualizadas$(NC)"

status: environment ## 📊 Alias para 'environment'

.DEFAULT_GOAL := help

################################################################################
# FIM DO MAKEFILE
################################################################################
