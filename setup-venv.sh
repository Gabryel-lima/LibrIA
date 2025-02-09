#!/bin/bash

# Nome do ambiente virtual
VENV_NAME=".venv"

# Verifica se o Python3 está instalado
if ! command -v python3 &> /dev/null; then
    echo "Python3 não encontrado. Instale o Python3 antes de continuar."
    exit 1
fi

# Cria o ambiente virtual
echo "Criando ambiente virtual $VENV_NAME..."
python3 -m venv $VENV_NAME

# Ativa o ambiente virtual
echo "Ativando o ambiente virtual..."
source $VENV_NAME/bin/activate

# Atualiza o pip dentro do ambiente virtual
echo "Atualizando pip..."
pip install --upgrade pip

# Instala pacotes essenciais (opcional)
echo "Instalando pacotes essenciais..."
pip install wheel setuptools

# Mensagem de sucesso
echo "Ambiente virtual criado e pronto para uso! Execute 'source $VENV_NAME/bin/activate' para ativá-lo."

# Mantém o shell ativo para evitar que o ambiente se desative imediatamente
exec "$SHELL"
