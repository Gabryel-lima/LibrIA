#!/bin/bash

# Script de configuração do ambiente para o projeto com pipenv
# Deve ser executado com permissões de superusuário (sudo).

# Em caso de erro, o script para imediatamente
set -e

# Função de utilidade para verificar se um comando está instalado
isCommandInstalled() {
    if [ $# -ne 1 ]; then 
        echo "Número inválido de parâmetros fornecido para $FUNCNAME" 
        return 1
    fi

    if ! command -v "$1" &> /dev/null; then 
        echo "$1 não está instalado" 
        return 1
    fi
}

# Verifica se o script está sendo executado como superusuário
if [ "$EUID" -ne 0 ]; then
    echo "❌ Este script deve ser executado com permissões de superusuário (sudo)."
    exit 1
fi

# Atualiza os pacotes do sistema
echo "🔄 Atualizando pacotes do sistema..."
sudo apt update && sudo apt upgrade -y

# Instala dependências do sistema e o Python 3.11
if ! isCommandInstalled python3.11 ; then
    echo "🐍 Instalando Python 3.11 e dependências..."
    sudo apt install -y software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    sudo apt install -y python3.11 python3.11-venv python3.11-distutils python3-pip python3-dev build-essential
else
    echo "🐍 Python 3.11 já instalado. Verificando dependências internas..."

    if ! python3.11 -m venv --help >/dev/null 2>&1; then
        echo "⚙️  Problema detectado no módulo venv. Reinstalando dependências..."
        sudo apt install --reinstall -y python3.11-venv python3.11-distutils python3-pip
    else
        echo "✅ venv funcionando corretamente."
    fi

    if ! python3.11 -m ensurepip --version >/dev/null 2>&1; then
        echo "⚙️  Problema detectado no ensurepip. Reinstalando dependências..."
        sudo apt install --reinstall -y python3.11-venv python3.11-distutils python3-pip
    else
        echo "✅ ensurepip funcionando corretamente."
    fi
fi

# Configura o ambiente virtual com pipenv e instala as dependências Python
echo "🐍 Configurando ambiente virtual com pipenv..."

# Verifica se o pipenv está instalado; caso não esteja, instala-o via pip
if ! isCommandInstalled pipenv ; then
    echo "🐍 Pipenv não encontrado. Instalando pipenv..."
    python3.11 -m pip install --user pipenv
    export PATH="$HOME/.local/bin:$PATH"
fi

# Verifica se há um arquivo requirements.txt
if ! [ -f "requirements.txt" ]; then
    echo "❌ Arquivo requirements.txt não encontrado. Abortando instalação."
    exit 1
fi

# Inicializa o pipenv com o interpretador Python 3.11 caso o Pipfile não exista
if [ ! -f "Pipfile" ]; then
    PY311_PATH=$(command -v python3.11)
    if [ -z "$PY311_PATH" ]; then
        echo "❌ Python 3.11 não encontrado no PATH. Abortando."
        exit 1
    fi
    echo "📌 Usando interpretador localizado em: $PY311_PATH"
    pipenv --python "$PY311_PATH"
fi

# Instala as dependências do requirements.txt no ambiente pipenv
pipenv install -r requirements.txt

# OBSERVAÇÃO IMPORTANTE
echo "⚠️  ATENÇÃO: NÃO ALTERAR o requirements.txt manualmente sem revalidar as dependências."
echo "⚠️  As versões dos pacotes estão travadas para garantir compatibilidade."

# Instruções finais
echo ""
echo "✅ Setup concluído com sucesso!"
echo "👉 Ative seu ambiente pipenv utilizando:"
echo "👉 pipenv shell"
echo "ℹ️ Após ativar o ambiente, se necessário, instale manualmente a biblioteca TTS para evitar conflitos com o numpy."
echo ""
