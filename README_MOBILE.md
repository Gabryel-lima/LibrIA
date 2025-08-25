# LibrIA Mobile - Aplicativo de Reconhecimento de Libras

## 📱 Sobre o Projeto Mobile

Este é o aplicativo mobile do **LibrIA**, desenvolvido em Flutter para Android e iOS. O app permite reconhecimento de Libras em tempo real usando a câmera do dispositivo.

## 🏗️ Arquitetura do Projeto

```
LibrIA/
├── 📱 mobile_app/                    # Aplicativo Flutter
│   ├── lib/
│   │   ├── core/                     # Serviços e utilitários
│   │   ├── features/                 # Funcionalidades do app
│   │   │   ├── camera/              # Configurações da câmera
│   │   │   ├── learning/            # Tela de aprendizado
│   │   │   └── recognition/         # Reconhecimento principal
│   │   └── shared/                  # Widgets e modelos compartilhados
│   ├── assets/                      # Recursos do app
│   └── pubspec.yaml                 # Dependências Flutter
├── 🔧 backend/                      # API FastAPI
│   ├── main.py                      # Servidor principal
│   └── requirements.txt             # Dependências Python
└── ☁️ cloud/                        # Configurações de nuvem (futuro)
```

## 🚀 Configuração do Ambiente

### Pré-requisitos

1. **Flutter SDK** (versão 3.0.0 ou superior)
2. **Android Studio** ou **VS Code**
3. **Python 3.8+** (para o backend)
4. **Git**

### 1. Instalar Flutter

```bash
# Clone o Flutter SDK
git clone https://github.com/flutter/flutter.git
export PATH="$PATH:`pwd`/flutter/bin"

# Verificar instalação
flutter doctor
```

### 2. Configurar Android Studio

1. Instalar Android Studio
2. Instalar Android SDK
3. Configurar emulador Android ou conectar dispositivo físico
4. Instalar plugin Flutter no Android Studio

### 3. Configurar Backend

```bash
# Navegar para o diretório do backend
cd backend

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt

# Verificar se o modelo existe
ls ../model/model.pickle
```

## 📱 Executando o App Mobile

### 1. Configurar Dependências

```bash
# Navegar para o app mobile
cd mobile_app

# Instalar dependências Flutter
flutter pub get

# Verificar dispositivos disponíveis
flutter devices
```

### 2. Executar o Backend

```bash
# Em um terminal separado
cd backend
python main.py
```

O backend estará disponível em: `http://localhost:8000`

### 3. Executar o App

```bash
# No diretório mobile_app
flutter run
```

## 🔧 Funcionalidades Implementadas

### ✅ Backend (FastAPI)

- [x] **API REST** para reconhecimento de Libras
- [x] **Endpoint `/predict`** para predição única
- [x] **Endpoint `/predict/batch`** para predição em lote
- [x] **Integração** com modelo existente
- [x] **Documentação automática** (Swagger UI)
- [x] **CORS configurado** para mobile
- [x] **Tratamento de erros** robusto

### ✅ App Flutter (Estrutura)

- [x] **Projeto configurado** com dependências
- [x] **Navegação** entre telas
- [x] **Tema** Material Design 3
- [x] **Estrutura de pastas** organizada
- [x] **Serviços** de permissões e logger
- [x] **Telas básicas** implementadas

### 🚧 Em Desenvolvimento

- [ ] **Integração com câmera** em tempo real
- [ ] **Comunicação com API** backend
- [ ] **Interface de usuário** completa
- [ ] **Funcionalidades de aprendizado**
- [ ] **Armazenamento local** de dados
- [ ] **Testes automatizados**

## 📋 Endpoints da API

### Base URL: `http://localhost:8000`

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/` | GET | Informações da API |
| `/health` | GET | Status de saúde |
| `/predict` | POST | Reconhecimento de letra |
| `/predict/batch` | POST | Reconhecimento em lote |
| `/model/info` | GET | Informações do modelo |
| `/alphabet` | GET | Alfabeto suportado |
| `/docs` | GET | Documentação Swagger |

### Exemplo de Uso da API

```bash
# Testar API
curl -X GET "http://localhost:8000/health"

# Fazer predição (requer imagem)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@sua_imagem.jpg"
```

## 🎯 Próximos Passos

### Fase 1: Integração Core (1-2 semanas)
1. **Implementar câmera** em tempo real
2. **Conectar com API** backend
3. **Processar imagens** e enviar para predição
4. **Exibir resultados** em tempo real

### Fase 2: Interface Completa (2-3 semanas)
1. **Melhorar UI/UX** das telas
2. **Adicionar animações** e feedback visual
3. **Implementar modo de aprendizado**
4. **Adicionar configurações** do usuário

### Fase 3: Funcionalidades Avançadas (3-4 semanas)
1. **Histórico** de reconhecimentos
2. **Modo offline** (modelo local)
3. **Jogos educativos**
4. **Sincronização** na nuvem

## 🐛 Solução de Problemas

### Problemas Comuns

| Problema | Solução |
|----------|---------|
| **Flutter não encontrado** | Adicionar ao PATH: `export PATH="$PATH:/caminho/para/flutter/bin"` |
| **Dependências não instaladas** | Executar `flutter pub get` |
| **Backend não inicia** | Verificar se modelo existe em `../model/model.pickle` |
| **Câmera não funciona** | Verificar permissões no dispositivo |
| **API não responde** | Verificar se backend está rodando na porta 8000 |

### Logs e Debug

```bash
# Logs do Flutter
flutter logs

# Logs do Backend
tail -f backend/libras.log

# Verificar dispositivos
flutter devices
```

## 📱 Preparação para Play Store

### Requisitos Técnicos

- [ ] **Target API**: Android 6.0+ (API 23)
- [ ] **Tamanho**: <100MB
- [ ] **Performance**: <2s carregamento
- [ ] **Permissões**: Câmera, Internet, Armazenamento

### Checklist de Lançamento

- [ ] **Ícone do app** (512x512)
- [ ] **Screenshots** de todas as telas
- [ ] **Descrição** completa
- [ ] **Política de privacidade**
- [ ] **Testes** em dispositivos reais
- [ ] **Otimização** de performance

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📞 Suporte

- 📧 Email: gabbryellimasi@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/Gabryel-lima/LibrIA/issues)
- 💬 Discussões: [GitHub Discussions](https://github.com/Gabryel-lima/LibrIA/discussions)

---

**LibrIA Mobile** - Tornando a comunicação em Libras mais acessível! 🤟
