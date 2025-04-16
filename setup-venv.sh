#!/bin/bash

set -e  # Encerra o script ao primeiro erro

# Script de configuração do ambiente para o projeto com pipenv
# Deve ser executado SEM permissões de superusuário (sudo).
# Apenas chmod +x arquivo.sh e execute o script.

# 🔐 Verifica se é superusuário para não executar
if [ "$EUID" -ne 0 ]; then
    echo "Por favor, execute como root apenas essa parte: sudo apt install pipenv"
    exit -1
fi

# 🔄 Atualiza os pacotes do sistema
echo "🔄 Atualizando pacotes do sistema..."
apt update && apt upgrade -y
echo "Trava padrão. Não UTILIZAR SUDO AQUI."
echo ""

# Nome do ambiente virtual
VENV_NAME=".venv"
VENV_PATH="./$VENV_NAME"
ACTIVATE_SCRIPT="$VENV_PATH/bin/activate"

# Função para mensagens coloridas
info() { echo -e "\033[1;34m[INFO]\033[0m $1"; }
error() { echo -e "\033[1;31m[ERRO]\033[0m $1"; }

# Verifica se o Python3 está instalado
if ! command -v python3 &> /dev/null; then
    error "Python3 não encontrado. Instale o Python3 antes de continuar."
    exit 1
fi

# Verifica se o ambiente virtual já existe
if [ -d "$VENV_PATH" ]; then
    info "O ambiente virtual '$VENV_NAME' já existe. Pulando criação..."
else
    info "Criando ambiente virtual '$VENV_NAME'..."
    python3 -m venv "$VENV_NAME"
fi

# Verifica se o script está sendo executado como superusuário
if [ "$EUID" -ne 0 ]; then
    echo "❌ Este script deve ser executado com permissões de superusuário (sudo)."
    exit 1
fi

# Ativa o ambiente virtual
info "Ativando o ambiente virtual..."
# shellcheck source=/dev/null
source "$ACTIVATE_SCRIPT"

# Atualiza o pip dentro do ambiente virtual
info "Atualizando pip..."
python -m pip install --upgrade pip

# Instala pacotes essenciais (opcional)
info "Instalando pacotes essenciais..."
pip install --upgrade wheel setuptools

info "Ambiente virtual pronto para uso!"
echo "Para ativar manualmente depois, execute: source $ACTIVATE_SCRIPT"

# Mantém o shell ativo se estiver em terminal interativo
if [[ $- == *i* ]]; then
    exec "$SHELL"
fi
