#!/bin/bash

# TODO: Não está completamente funcional, mas é um bom ponto de partida
# Deve procorar o Pipfile e Pipfile.lock na pasta atual
# e criar um ambiente virtual com o Pipenv, instalando as dependências

set -e  # Encerra o script ao primeiro erro

# Função para mensagens coloridas
info() { echo -e "\033[1;34m[INFO]\033[0m $1"; }
error() { echo -e "\033[1;31m[ERRO]\033[0m $1"; }

# Verifica se o Python3 está instalado
if ! command -v python3 &> /dev/null; then
    error "Python3 não encontrado. Instale o Python3 antes de continuar."
    exit 1
fi

# Verifica se o pipenv está instalado
if ! command -v pipenv &> /dev/null; then
    error "Pipenv não encontrado. Instale o Pipenv antes de continuar."
    exit 1
fi

# Inicializa o ambiente virtual com pipenv
info "Inicializando o ambiente virtual com Pipenv..."
pipenv --python 3

# Atualiza o pip no ambiente virtual
info "Atualizando pip no ambiente virtual..."
pipenv run python -m pip install --upgrade pip

# Instala pacotes essenciais (opcional)
info "Instalando pacotes essenciais..."
pipenv install wheel setuptools --dev

info "Ambiente virtual gerenciado pelo Pipenv pronto para uso!"
echo "Para ativar o ambiente, utilize: pipenv shell"
