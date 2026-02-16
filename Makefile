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

.PHONY: help setup install install-gpu install-cpu collect process train infer \
        run test clean clean-all dirs verify-setup environment lint format

# Variáveis de Configuração
PYTHON := python3
PYTHON_VERSION := 3.11
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
PROJECT_NAME := LibrIA
PROJECT_VERSION := 1.0.0

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
	@awk 'BEGIN {FS = ":.*?## "} /^run|^collect|^process|^train|^infer/ && !/^setup/ && !/^$$/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
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
	@echo "  $(CYAN)make setup          # Setup inicial"
	@echo "  make run            # Executar pipeline completo"
	@echo "  make infer          # Inferência em tempo real$(NC)"
	@echo ""

setup: $(VENV_DIR) ## 🔧 Setup inicial - cria venv e instala dependências
	@echo "$(GREEN)✓ Ambiente virtual criado em $(VENV_DIR)$(NC)"
	@echo "$(BLUE)→ Instalando dependências ($(DEFAULT_INSTALL))...$(NC)"
	@$(MAKE) --no-print-directory $(DEFAULT_INSTALL)
	@echo "$(GREEN)✓ Setup completo! Use 'make verify-setup' para validar.$(NC)"

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
	@echo "$(GREEN)✓ MediaPipe: $(shell $(VENV_PYTHON) -c 'import mediapipe; print(mediapipe.__version__)' 2>/dev/null)$(NC)"
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
# PIPELINE - COLLECT, PROCESS, TRAIN, INFER
################################################################################

dirs: ## 📂 Cria a estrutura de diretórios do projeto
	@echo "$(BLUE)→ Criando estrutura de diretórios...$(NC)"
	@mkdir -p data dataset model output training_plots
	@echo "$(GREEN)✓ Diretórios criados: data/, dataset/, model/, output/, training_plots/$(NC)"

collect: verify-setup dirs ## 📷 Coleta dados com webcam (todas as classes)
	@echo "$(YELLOW)→ Iniciando coleta de dados para todas as classes...$(NC)"
	@echo "$(MAGENTA)  Pressione 'q' para sair de cada classe$(NC)"
	@$(VENV_PYTHON) main.py collect

collect-jz: verify-setup dirs ## 📷 Coleta dados apenas para J e Z
	@echo "$(YELLOW)→ Coletando dados específicos (J e Z)...$(NC)"
	@$(VENV_PYTHON) collect_j_z.py

process: verify-setup ## ⚙️  Processa dados - extrai landmarks
	@echo "$(MAGENTA)→ Processando dataset (extração de landmarks)...$(NC)"
	@$(VENV_PYTHON) main.py process
	@echo "$(GREEN)✓ Processamento completo! Dataset em ./dataset/data.pickle$(NC)"

train: verify-setup ## 🤖 Treina modelo Random Forest
	@echo "$(MAGENTA)→ Iniciando treinamento do modelo...$(NC)"
	@$(VENV_PYTHON) main.py train
	@echo "$(GREEN)✓ Modelo treinado! Salvo em ./model/model.pickle$(NC)"

infer: verify-setup ## 🎯 Inferência em tempo real com webcam
	@echo "$(GREEN)→ Iniciando inferência em tempo real...$(NC)"
	@echo "$(MAGENTA)  Pressione 'q' para sair$(NC)"
	@$(VENV_PYTHON) main.py infer

run: verify-setup dirs ## ▶️  Executa pipeline completo (collect→process→train→infer)
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
