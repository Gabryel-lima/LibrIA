#!/bin/bash

# LibrIA Mobile Development Script
# Este script facilita o desenvolvimento do projeto mobile

echo "🚀 LibrIA Mobile - Script de Desenvolvimento"
echo "=============================================="

# Função para verificar se um comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Verificar pré-requisitos
echo "📋 Verificando pré-requisitos..."

if ! command_exists python3; then
    echo "❌ Python 3 não encontrado. Instale o Python 3.8+"
    exit 1
fi

if ! command_exists flutter; then
    echo "❌ Flutter não encontrado. Instale o Flutter SDK"
    echo "   https://flutter.dev/docs/get-started/install"
    exit 1
fi

echo "✅ Pré-requisitos verificados!"

# Função para configurar backend
setup_backend() {
    echo "🔧 Configurando Backend..."
    cd backend
    
    # Criar ambiente virtual se não existir
    if [ ! -d "venv" ]; then
        echo "📦 Criando ambiente virtual..."
        python3 -m venv venv
    fi
    
    # Ativar ambiente virtual
    source venv/bin/activate
    
    # Instalar dependências
    echo "📦 Instalando dependências Python..."
    pip install -r requirements.txt
    
    # Verificar se o modelo existe
    if [ ! -f "../model/model.pickle" ]; then
        echo "⚠️  Modelo não encontrado. Execute o treinamento primeiro:"
        echo "   cd .. && python main.py train"
        exit 1
    fi
    
    echo "✅ Backend configurado!"
    cd ..
}

# Função para configurar mobile app
setup_mobile() {
    echo "📱 Configurando App Mobile..."
    cd mobile_app
    
    # Instalar dependências Flutter
    echo "📦 Instalando dependências Flutter..."
    flutter pub get
    
    # Verificar dispositivos
    echo "📱 Dispositivos disponíveis:"
    flutter devices
    
    echo "✅ App Mobile configurado!"
    cd ..
}

# Função para iniciar backend
start_backend() {
    echo "🚀 Iniciando Backend..."
    cd backend
    source venv/bin/activate
    python main.py &
    BACKEND_PID=$!
    echo "✅ Backend iniciado (PID: $BACKEND_PID)"
    cd ..
}

# Função para iniciar app mobile
start_mobile() {
    echo "📱 Iniciando App Mobile..."
    cd mobile_app
    
    # Verificar se há dispositivos conectados
    DEVICES=$(flutter devices | grep -c "device")
    if [ "$DEVICES" -eq 0 ]; then
        echo "❌ Nenhum dispositivo encontrado. Conecte um dispositivo ou inicie um emulador."
        exit 1
    fi
    
    flutter run
    cd ..
}

# Função para parar todos os processos
cleanup() {
    echo "🧹 Limpando processos..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        echo "✅ Backend parado"
    fi
    exit 0
}

# Capturar Ctrl+C para limpeza
trap cleanup SIGINT

# Menu principal
show_menu() {
    echo ""
    echo "🎯 Escolha uma opção:"
    echo "1) Configurar tudo (Backend + Mobile)"
    echo "2) Configurar apenas Backend"
    echo "3) Configurar apenas Mobile"
    echo "4) Iniciar Backend"
    echo "5) Iniciar App Mobile"
    echo "6) Iniciar tudo (Backend + Mobile)"
    echo "7) Verificar status"
    echo "8) Sair"
    echo ""
    read -p "Digite sua opção (1-8): " choice
}

# Função para verificar status
check_status() {
    echo "📊 Status do Projeto:"
    echo ""
    
    # Verificar backend
    if [ -d "backend/venv" ]; then
        echo "✅ Backend: Ambiente virtual configurado"
    else
        echo "❌ Backend: Ambiente virtual não configurado"
    fi
    
    # Verificar modelo
    if [ -f "model/model.pickle" ]; then
        echo "✅ Modelo: Treinado e disponível"
    else
        echo "❌ Modelo: Não encontrado"
    fi
    
    # Verificar mobile
    if [ -d "mobile_app" ]; then
        echo "✅ Mobile: Projeto Flutter configurado"
    else
        echo "❌ Mobile: Projeto não encontrado"
    fi
    
    # Verificar dispositivos
    echo ""
    echo "📱 Dispositivos disponíveis:"
    cd mobile_app
    flutter devices
    cd ..
}

# Loop principal
while true; do
    show_menu
    
    case $choice in
        1)
            setup_backend
            setup_mobile
            ;;
        2)
            setup_backend
            ;;
        3)
            setup_mobile
            ;;
        4)
            start_backend
            echo "Backend rodando em http://localhost:8000"
            echo "Pressione Ctrl+C para parar"
            wait $BACKEND_PID
            ;;
        5)
            start_mobile
            ;;
        6)
            setup_backend
            setup_mobile
            start_backend
            sleep 3
            start_mobile
            ;;
        7)
            check_status
            ;;
        8)
            echo "👋 Até logo!"
            exit 0
            ;;
        *)
            echo "❌ Opção inválida. Tente novamente."
            ;;
    esac
    
    echo ""
    read -p "Pressione Enter para continuar..."
done
