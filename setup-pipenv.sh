#!/bin/bash

# Script de configuração do ambiente para o projeto com pipenv
# Deve ser executado SEM permissões de superusuário (sudo).
# Apenas chmod +x arquivo.sh e execute o script.

set -e  # Para o script imediatamente em caso de erro

# 🔐 Verifica se é superusuário para não executar
if [ "$EUID" -ne 0 ]; then
    echo "Por favor, execute como root apenas essa parte: sudo apt install pipenv"
    exit -1
fi

# 🧰 Função auxiliar
# 🔐 Verifica se é superusuário para executar
# if [ "$EUID" -ne 0 ]; then
#     echo "❌ Este script deve ser executado com sudo."
#     exit 1
# fi

# 🧰 Função auxiliar para verificar comandos
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

# 🔄 Atualiza os pacotes do sistema
echo "🔄 Atualizando pacotes do sistema..."
apt update && apt upgrade -y
echo "Trava padrão. Não UTILIZAR SUDO AQUI."
echo ""

# 🐍 Instala o Python 3.11 se necessário
if ! isCommandInstalled python3.11 ; then
    echo "🐍 Instalando Python 3.11 e dependências..."
    apt install -y software-properties-common
    add-apt-repository ppa:deadsnakes/ppa -y
    apt update
    apt install -y python3.11 python3.11-venv python3.11-distutils python3-pip python3-dev build-essential
else
    echo "🐍 Python 3.11 já instalado. Verificando dependências internas..."

    if ! python3.11 -m venv --help >/dev/null 2>&1; then
        echo "⚙️  Reinstalando venv e distutils..."
        apt install --reinstall -y python3.11-venv python3.11-distutils python3-pip
    else
        echo "✅ venv funcionando corretamente."
    fi

    if ! python3.11 -m ensurepip --version >/dev/null 2>&1; then
        echo "⚙️  Reinstalando ensurepip..."
        apt install --reinstall -y python3.11-venv python3.11-distutils python3-pip
    else
        echo "✅ ensurepip funcionando corretamente."
    fi
fi

# 📦 Instala pipenv (forçando reinstalação caso corrompido)
echo "🐍 Instalando pipenv no ambiente do usuário..."
export PATH="$HOME/.local/bin:$PATH"
if command -v pipenv &> /dev/null; then
    echo "🔁 Reinstalando pipenv (force)..."
    python3.11 -m pip install --user --force-reinstall pipenv
else
    echo "➕ Instalando pipenv..."
    python3.11 -m pip install --user pipenv
fi

# 📄 Verifica o arquivo requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "❌ Arquivo requirements.txt não encontrado."
    exit 1
fi

# 🧹 Remove ambiente antigo se existir
if [ -f "Pipfile.lock" ]; then
    echo "🧹 Limpando ambiente pipenv anterior..."
    pipenv --rm || echo "ℹ️ Ambiente não existia."
    rm -f Pipfile.lock
fi

# ⚙️ Cria novo ambiente com python3.11 absoluto
PY311_PATH=$(command -v python3.11)
if [ -z "$PY311_PATH" ]; then
    echo "❌ Python 3.11 não localizado no PATH. Abortando."
    exit 1
fi
echo "📌 Usando interpretador localizado em: $PY311_PATH"
pipenv --python "$PY311_PATH"

# 📥 Instala dependências do projeto
echo "📦 Instalando dependências do requirements.txt..."
pipenv install -r requirements.txt

# 📌 Finalização
echo ""
echo "✅ Setup concluído com sucesso!"
echo "👉 Ative o ambiente com: pipenv shell"
echo "ℹ️ Após ativar, instale manualmente bibliotecas específicas se necessário (ex: TTS, numpy)."
echo ""
