# LibrIA - Reconhecimento de Libras com Visão Computacional

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

</div>

## 📖 Sobre o Projeto

O **LibrIA** é um sistema de reconhecimento de Libras baseado em visão computacional, com fluxos host e embedded já integrados ao código. O repositório cobre coleta unificada em `dataset/`, treino, inferência em tempo real, calibração opcional de câmera e export de um bundle quantizado com pacote pronto para o Pico.

### 🎯 Objetivos

- **Acessibilidade**: Facilitar a comunicação entre pessoas surdas e ouvintes
- **Educação**: Apoiar o aprendizado de Libras
- **Tecnologia**: Demonstrar aplicações práticas de IA em visão computacional
- **Inovação**: Criar soluções inclusivas usando machine learning

### ✨ Características

- 🎥 **Captura em tempo real** via webcam
- 🤖 **Fluxos host e embedded**: Random Forest, LSTM, CNN estática quantizada e CNN temporal quantizada
- 📊 **Pipeline completo** de dados estáticos e temporais
- 🎯 **Inferência contínua** com feedback visual
- 📷 **Calibração opcional de câmera** com tabuleiro 9x6
- 📦 **Bundle embedded exportável** com pacote C/C++ para Pico
- 🏗️ **Arquitetura modular** e configurável por `FEATURE_MODE`

## 📚 Documentação Completa

**👉 [Índice Completo de Documentação](DOCUMENTATION_INDEX.md)** - Referência rápida para encontrar o que você precisa

### 🚀 Comece Aqui
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Como contribuir para o projeto
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** - Setup de desenvolvimento local
- **[docs/PULL_REQUEST_GUIDE.md](docs/PULL_REQUEST_GUIDE.md)** - Guia de Pull Requests

### 📖 Documentação do Projeto
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Arquitetura visual com diagramas curtos e navegáveis
- **[docs/DATASETS.md](docs/DATASETS.md)** - Datasets disponíveis e como obter dados
- **[docs/AVX_COMPATIBILITY.md](docs/AVX_COMPATIBILITY.md)** - Guia para CPUs sem suporte AVX
- **[docs/video_format_changes.md](docs/video_format_changes.md)** - Mudanças de formatos de vídeo

### 🏛️ Governança e Comunidade
- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** - Código de conduta
- **[GOVERNANCE.md](GOVERNANCE.md)** - Estrutura de governança e decisões
- **[CONTRIBUTORS.md](CONTRIBUTORS.md)** - Lista de contribuidores
- **[ROADMAP.md](ROADMAP.md)** - Visão futura do projeto
- **[CHANGELOG.md](CHANGELOG.md)** - Histórico de mudanças

### 📋 Outros
- **[LICENSE](LICENSE)** - Licença MIT
## 🛠️ Tecnologias Utilizadas

| Tecnologia | Versão | Propósito |
|------------|--------|-----------|
| **Python** | 3.11+ | Linguagem principal |
| **TensorFlow** | 2.16.1 | Deep Learning para o pipeline temporal |
| **Keras** | 3.4.1 | API de alto nível |
| **PyTorch** | 2.5.1 | Alternativa de Deep Learning |
| **OpenCV** | 4.11+ | Captura e processamento de vídeo |
| **MediaPipe** | 0.10+ | Detecção de landmarks da mão |
| **Scikit-learn** | 1.6+ | Random Forest (modelo principal) |
| **NumPy** | 1.26+ | Computação numérica |
| **Pandas** | 2.2+ | Manipulação de dados |
| **Matplotlib** | 3.10+ | Visualização |
| **Serialização** | pickle | Salvar modelos e dados |

## 📁 Estrutura Atual do Projeto

```
LibrIA/
├── 📁 src/                          # Código principal
│   ├── 📁 model_training/           # Treino Random Forest e LSTM
│   ├── 📁 inference/                # Inferência estática e temporal
│   └── 📁 models/                   # Modelos/experimentos adicionais
├── 📁 scripts/                      # Utilitários de calibração e coleta unificada
├── 📁 config/                       # Configurações centrais e arquivos de calibração
├── 📁 utils/                        # Helpers compartilhados
├── 📁 dataset/                      # Dataset unificado
│   ├── static/
│   └── temporal/
├── 📁 model/                        # Artefatos treinados
│   ├── model.pickle
│   ├── libras_lstm.keras
│   ├── libras_lstm_labels.pickle
│   └── embedded_bundle/
├── 📁 docs/                         # Documentação detalhada
├── 📄 Makefile                      # Automação de setup e execução
├── 📄 main.py                       # Interface principal por comandos
├── 📄 requirements.txt              # Dependências principais
├── 📄 requirements-dev.txt          # Dependências de desenvolvimento
├── 📄 test_setup.py                 # Verificação de ambiente
└── 📄 README.md                     # Visão geral do projeto
```

## 🚀 Instalação e Configuração

### ⚡ Quick Start (Recomendado)

A forma mais rápida de começar é usando o **Makefile**:

```bash
# 1. Clone o repositório
git clone https://github.com/Gabryel-lima/LibrIA.git
cd LibrIA

# 2. Setup inicial (cria venv e instala dependências)
make setup

# 3. Verifique se tudo está funcionando
make verify-setup

# 4. Veja todos os comandos disponíveis
make help
```

### Pré-requisitos

- **Python 3.11** ou superior
- **Webcam funcional**
- **Git**
- **CPU com suporte AVX** (recomendado para funcionalidades completas)
  - ℹ️ Sem suporte AVX? Veja [Compatibilidade AVX](docs/AVX_COMPATIBILITY.md)
- **CUDA/cuDNN** (opcional, para GPU)

### Setup Manual (Alternativa)

Se preferir configurar manualmente sem o Makefile:

```bash
# Clone o repositório
git clone https://github.com/Gabryel-lima/LibrIA.git
cd LibrIA

# Crie um ambiente virtual
python3.11 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# Instale as dependências (CPU)
pip install -r requirements.txt

# Ou para GPU (requer CUDA 12.4):
pip install -r requirements.txt -r requirements-gpu.txt

# Verifique a instalação
python -c "import cv2, mediapipe, sklearn; print('✓ Dependências instaladas!')"
```

### Verificação de Setup

Use o comando para validar todo o ambiente:

```bash
make verify-setup
```

Ou manualmente:
```bash
python test_setup.py
```

## 📊 Como Usar

### 🎯 Makefile - Comandos Disponíveis

O projeto agora utiliza **Makefile** para automatizar todas as operações:

```bash
# Ver todos os comandos e status do sistema
make help

# Setup inicial
make setup                  # Cria venv + instala dependências
make verify-setup          # Valida o ambiente
make environment           # Mostra informações do sistema

# Pipeline de ML
make dirs                  # Cria estrutura de diretórios
make collect-static       # Coleta dataset estático unificado
make collect-temporal     # Coleta dataset temporal unificado
make collect-minimal-dataset # Coleta o dataset mínimo completo
make train                # Treina modelo Random Forest
make train-lstm           # Treina modelo temporal LSTM
make train-embedded       # Treina CNN estática quantizada
make train-embedded-temporal # Treina CNN temporal quantizada para J/Z
make train-embedded-all   # Treina os dois modelos embedded e exporta o bundle
make export-embedded      # Reempacota bundle e pacote do Pico
make train-hybrid         # Retreina os dois modelos
make infer                # Inferência em tempo real
make infer-lstm           # Inferência temporal em tempo real
make infer-hybrid         # Inferência híbrida
make infer-embedded       # Verifica o bundle embedded com os datasets NPY
make run                  # Executa pipeline completo unificado
make run-lstm             # Executa pipeline temporal completo

# Desenvolvimento
make test                 # Executa testes
make lint                 # Verifica código (requer dev dependencies)
make format               # Formata código (requer dev dependencies)
make install-dev          # Instala dependências de desenvolvimento

# Utilitários
make clean                # Remove __pycache__, arquivos temporários

# Calibração de câmera
make generate-checkerboard  # Gera a imagem do tabuleiro 9x6
make show-checkerboard      # Exibe o tabuleiro gerado em tela cheia
make capture-calibration  # Abre a webcam, salva imagens do tabuleiro e calibra
make calibrate-camera CALIBRATION_IMAGES='calibration/*.jpg'

make clean-all            # Remove tudo (venv, dados, modelos) - CUIDADO!
make freeze               # Gera requirements.lock com versões exatas
make update               # Atualiza dependências
make status               # Alias para 'environment'
```

Para calibrar a câmera, use um tabuleiro de xadrez com 9x6 cantos internos.
O fluxo recomendado e:
1. Rodar make generate-checkerboard.
2. Imprimir a imagem gerada ou exibi-la em outra tela com make show-checkerboard.
3. Rodar make capture-calibration.
4. Mover o tabuleiro em distâncias e ângulos diferentes.
5. Pressionar espaco quando o status mostrar que o tabuleiro foi detectado.
6. A calibração sera salva em config/camera_matrix.npy e config/dist_coeffs.npy.

Importante: a janela da webcam não desenha o tabuleiro automaticamente. A câmera precisa ver fisicamente a imagem impressa ou a tela com o tabuleiro aberto.

### ⚙️ Detecção Automática

O Makefile detecta automaticamente:
- ✅ **CUDA/GPU**: Se disponível, instala com suporte GPU (CUDA 12.4)
- ✅ **CPU**: Fallback automático para CPU-only
- ✅ **Python**: Usa Python 3.11+

### 🎯 Interface Unificada (main.py)

Você também pode usar o script `main.py` diretamente:

```bash
# Mostrar ajuda
python main.py help

# Executar pipeline completo
python main.py all

# Executar etapas individuais
python main.py collect_static   # Coleta estática
python main.py collect_temporal # Coleta temporal
python main.py train            # Treinar modelo
python main.py train_hybrid     # Retreino estático + temporal com status code confiável
python main.py infer            # Inferência em tempo real
python main.py infer_hybrid     # Inferência híbrida com arbitragem
python main.py train_embedded_all # Treino embedded completo
python main.py export_embedded    # Bundle e pacote do Pico
python main.py infer_embedded     # Validação do bundle embedded
```

Para automação local, CI ou scripts de shell, `main.py` agora propaga sucesso ou falha via exit code: `0` quando o comando conclui e `1` quando alguma etapa retorna erro.

## 🧭 Arquitetura Visual

Se você quiser entender o projeto por etapas, sem um diagrama único gigante, use:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

Esse guia separa o fluxo em blocos menores:

- entradas e comandos
- coleta e dataset
- treino host
- inferência host
- pipeline embedded
- export para Pico
- runtime no dispositivo

Cada diagrama termina com links para a sequência correta de leitura.

### 🔄 Pipeline Completo

Para executar todo o pipeline de uma vez:

```bash
python main.py all
```

Este comando irá:
1. ✅ Coletar o dataset mínimo em `dataset/`
2. ✅ Treinar os modelos estático e temporal
3. ✅ Executar inferência híbrida

### 📝 Instruções Detalhadas

#### 1. Fluxo estático

```bash
make collect-static
make train
make infer
```

O fluxo estático grava `frame_XXX.png` e `sample_XXX.npy` em `dataset/static/<label>/` e treina um Random Forest salvo em `model/model.pickle`.

Durante a coleta estática, o projeto também grava `sample_XXX_mirror.npy` para espelhar a amostra no eixo X. Se a pasta da classe já tiver apenas `frame_XXX.png`, a coleta faz backfill automático dos `sample_XXX.npy` ausentes antes de abrir a captura ao vivo e gera as versões espelhadas junto com as originais.

#### 1.1. Fluxo embedded estático sem MediaPipe

```bash
make collect-static
make train-embedded
```

Esse fluxo reutiliza os `sample_XXX.npy` do dataset estático, treina uma CNN pequena sobre landmarks normalizados e exporta:

- `model/libria_embedded_cnn.keras`
- `model/libria_embedded_cnn_int8.tflite`
- `model/libria_embedded_cnn_labels.json`

Esse caminho foi pensado para deployment embedded e é o ponto de partida recomendado para portar letras estáticas para microcontroladores. Como a automação principal do projeto está baseada em `.npy`, essa trilha usa diretamente os landmarks já coletados.

#### 1.2. Fluxo embedded temporal para J e Z

```bash
make collect-temporal SEQUENCE_LABELS=J\ Z SEQUENCE_COUNT=30 SEQUENCE_LENGTH=30
make train-embedded-temporal
```

Esse fluxo reutiliza `seq_XXX.npy` em `dataset/temporal/`, treina uma CNN temporal leve para capturar movimento e exporta:

- `model/libria_embedded_temporal_cnn.keras`
- `model/libria_embedded_temporal_cnn_int8.tflite`
- `model/libria_embedded_temporal_cnn_labels.json`

Para embedded, a estratégia recomendada passa a ser duas trilhas complementares:

- CNN estática para as letras sem movimento
- CNN temporal para `J` e `Z`

Se quiser treinar as duas em sequência:

```bash
make train-embedded-all
```

Isso já exporta automaticamente o bundle combinado em `model/embedded_bundle/`.

Se quiser apenas reempacotar os artefatos existentes:

```bash
make export-embedded
```

Para validar se o runtime híbrido está coerente com os `.npy` do projeto:

```bash
make infer-embedded
```

Esse comando não usa MediaPipe. Ele carrega o bundle quantizado, roda os modelos TFLite sobre `sample_XXX.npy` e `seq_XXX.npy` e reporta acurácia estática, temporal e híbrida.

O esqueleto de runtime no dispositivo ficou em:

- `src/interfaces/libria_embedded_runtime.h`
- `src/interfaces/libria_embedded_runtime.cpp`

Observação importante sobre o MediaPipe:

- a coleta do dataset continua igual e não foi alterada
- o MediaPipe segue apenas no host para produzir os landmarks do dataset atual
- o bundle embedded usa o mesmo contrato geométrico de landmarks, mas não depende do MediaPipe em runtime
- no Pico, o caminho viável é reproduzir o mesmo layout normalizado de landmarks com outro extrator leve ou firmware dedicado

#### 2. Fluxo temporal

```bash
make collect-temporal SEQUENCE_LABELS=J\ Z SEQUENCE_COUNT=30 SEQUENCE_LENGTH=30
make train-lstm
make infer-lstm
```

Esse fluxo grava sequências em `dataset/temporal/`, treina o modelo `model/libras_lstm.keras` e usa uma janela deslizante para inferência em tempo real.

Cada sequência válida também ganha uma contraparte `seq_XXX_mirror.npy`. No treino host e no treino temporal embedded, o carregador aceita o diretório temporal legado como fallback quando o caminho principal configurado existe, mas está vazio.

#### 2.1. Dataset mínimo recomendado

```bash
make collect-minimal-dataset
make train-hybrid
make infer-hybrid
```

Esse é o fluxo recomendado para validar a arquitetura com 24 classes estáticas e as sequências temporais de `J` e `Z`.

Na inferência híbrida, a etapa estática e a etapa temporal agora comparam a amostra original com sua versão espelhada e mantêm a hipótese de maior confiança antes da arbitragem final.

#### 3. Calibração de câmera

```bash
make generate-checkerboard
make show-checkerboard
make capture-calibration
```

Arquivos gerados:

- `config/camera_matrix.npy`
- `config/dist_coeffs.npy`

Esses parâmetros são usados de forma opcional no pré-processamento dos frames antes da extração dos landmarks.

#### 4. Extração de features

O modo de features é configurado em `config/settings.py`:

- `bounding_box`: 42 features
- `wrist_relative`: 63 features

O padrão atual é `wrist_relative`.

## 📥 Datasets e Downloads

O projeto trabalha com um dataset unificado em `dataset/`:

- **Estático**: `dataset/static/<label>/frame_XXX.png`, `sample_XXX.npy` e `sample_XXX_mirror.npy`
- **Temporal**: `dataset/temporal/<label>/seq_XXX.npy` e `seq_XXX_mirror.npy`

Recursos úteis:

- Dataset de apoio em `data/archives/`
- Modelos treinados em `model/`

Para detalhes de estrutura, formatos e artefatos, consulte [docs/DATASETS.md](docs/DATASETS.md).

## 🎯 Alfabeto Suportado

O sistema reconhece todas as 26 letras do alfabeto em Libras:

| Letra | Classe | Letra | Classe | Letra | Classe |
|-------|--------|-------|--------|-------|--------|
| A | 0 | I | 8  | R | 17|
| B | 1 | J | 9  | S | 18|
| C | 2 | K | 10 | T | 19|
| D | 3 | L | 11 | U | 20|
| E | 4 | M | 12 | V | 21|
| F | 5 | N | 13 | W | 22|
| G | 6 | O | 14 | X | 23|
| H | 7 | P | 15 | Y | 24|
|   |   | Q | 16 | Z | 25|

**Nota**: o fluxo estático recomendado usa 24 letras sem `J` e `Z`. Esses dois sinais ficam no pipeline temporal por dependerem de movimento.

## 🔬 Arquitetura Técnica

### Pipeline de Dados

#### Pipeline estático

1. **Captura**: webcam → OpenCV
2. **Detecção**: MediaPipe → landmarks da mão
3. **Features**: `bounding_box` ou `wrist_relative`
4. **Treinamento**: Random Forest
5. **Inferência**: classificação por frame

#### Pipeline temporal

1. **Captura**: webcam → OpenCV
2. **Pré-processamento**: calibração opcional da câmera
3. **Features**: sequência de vetores por frame
4. **Treinamento**: LSTM sobre janelas de 30 frames
5. **Inferência**: classificação com buffer deslizante

### Características dos Modelos

#### Random Forest

- **Algoritmo**: `RandomForestClassifier`
- **Features**: configuráveis por `FEATURE_MODE`
- **Persistência**: `model/model.pickle`
- **Metadados**: `feature_mode` e `num_features` salvos junto ao modelo

#### LSTM temporal

- **Sequência padrão**: 30 frames
- **Feature size padrão**: 63
- **Persistência**: `model/libras_lstm.keras`
- **Mapa de labels**: `model/libras_lstm_labels.pickle`

### 🚀 Modelos Alternativos em Desenvolvimento

O projeto também possui uma implementação de **Transformer-based models** para reconhecimento mais avançado. Para explorar modelos experimentais e futuras arquiteturas de deep learning, confira:

📁 **[src/models/transformer-gpt/](src/models/transformer-gpt/)** - Modelos Transformer e GPT em desenvolvimento

Estes modelos são uma alternativa em desenvolvimento ao Random Forest padrão, oferecendo potencial para:
- Reconhecimento temporal de sequências de gestos
- Melhor captura de nuances nos sinais
- Suporte a contexto e dependências de longo alcance

**📖 Documentação Completa**: [Leia o README detalhado](src/models/transformer-gpt/README.md) com instruções sobre:
- Dados para treinamento (interno e externo)
- Uso independente ou integrado ao projeto
- Exemplos de código prontos para usar
- Estratégias de integração com LibrIA

### Processamento de Imagens

```python
from utils.helpers import extract_landmarks_by_mode

results = hands.process(img_rgb)
if results.multi_hand_landmarks:
  features = extract_landmarks_by_mode(
    results.multi_hand_landmarks[0].landmark,
    FEATURE_MODE,
  )
```

## 📈 Artefatos e Saídas

Arquivos mais importantes gerados pelo pipeline:

- `dataset/static/<label>/sample_XXX.npy`
- `dataset/static/<label>/sample_XXX_mirror.npy`
- `dataset/static/<label>/frame_XXX.png`
- `dataset/temporal/<label>/seq_XXX.npy`
- `dataset/temporal/<label>/seq_XXX_mirror.npy`
- `model/model.pickle`
- `model/libras_lstm.keras`
- `model/libras_lstm_labels.pickle`
- `training_plots/training_history_lstm.png`
- `training_plots/training_history_embedded.png`
- `training_plots/training_history_temporal_embedded.png`
- `training_plots/accuracy/accuracy_*.png`
- `config/camera_matrix.npy`
- `config/dist_coeffs.npy`

## 🎥 Demonstração

[![Vídeo de Demonstração](https://img.shields.io/badge/YouTube-Demonstração-red)](https://youtu.be/WT0BZP_kRvQ?si=gDnGLICMUs_cUyn-)

## 🔧 Personalização

### Ajustando Parâmetros

Você pode modificar os parâmetros no arquivo `config/settings.py`:

```python
# Configurações de coleta
COLLECTION_CONFIG = {
  'static_samples_per_label': 30,
  'temporal_samples_per_label': 30,
}

# Configurações de inferência
INFERENCE_CONFIG = {
    'min_detection_confidence': 0.3,
    'prediction_interval': 20,  # Frames entre predições
}

# Configurações do modelo
MODEL_CONFIG = {
    'n_estimators': 100,
    'random_state': 42,
}
```

### Adicionando Novas Classes

1. Modifique `STATIC_LABELS` ou `TEMPORAL_LABELS` em `config/settings.py`
2. Rode a coleta correspondente com `python main.py collect_static` ou `python main.py collect_temporal`
3. Retreine com `python main.py train` ou `python main.py train_lstm`

## 🐛 Solução de Problemas

### Problemas Comuns

| Problema | Solução |
|----------|---------|
| **Webcam não detectada** | Verifique permissões e drivers |
| **Baixa acurácia** | Colete mais dados ou melhore a iluminação |
| **Detecção instável** | Ajuste `min_detection_confidence` |
| **Erro de importação** | Instale as dependências corretamente |
| **Módulo não encontrado** | Execute `python main.py` a partir da raiz do projeto |
| **`illegal hardware instruction (core dumped)`** | CPU sem suporte AVX - veja [Compatibilidade AVX](docs/AVX_COMPATIBILITY.md) |

### Logs e Debug

Para debug detalhado, verifique o arquivo `libras.log`:

```bash
tail -f libras.log
```

## 🏗️ Melhorias na Arquitetura

### ✅ O que foi melhorado:

1. **Makefile Profissional**: Automação de todas as tarefas (setup, run, test, clean)
2. **Dependências Consolidadas**: requirements.txt com versões exatas e suporte a GPU
3. **Estrutura Modular**: Código organizado em módulos específicos
4. **Configurações Centralizadas**: Todas as configurações em `config/settings.py`
5. **Utilitários Reutilizáveis**: Funções auxiliares em `utils/helpers.py`
6. **Interface Unificada**: Script principal `main.py` com comandos claros
7. **Documentação Melhorada**: Docstrings e comentários detalhados
8. **Tratamento de Erros**: Melhor gestão de exceções
9. **Logging**: Sistema de logs para debug
10. **Validação**: Verificação de pré-requisitos antes da execução
11. **Reproduzibilidade**: Lock files e CI/CD-ready
12. **Sem Pipenv**: Substituído por venv padrão do Python + Makefile

### 🔄 Migração dos Arquivos Antigos

Os arquivos e configurações obsoletas foram movidos para `.deprecated-files/`:
- `Pipfile` → `.deprecated-files/Pipfile` (pipenv descontinuado)
- `Pipfile.lock` → `.deprecated-files/Pipfile.lock` 
- `setup-pipenv.sh` → `.deprecated-files/setup-pipenv.sh` (obsoleto)
- `setup-venv.sh` → `.deprecated-files/setup-venv.sh` (substituído por Makefile)

Os arquivos de código antigos ainda estão em `backup_old_files/` apenas como referência histórica de experimentos anteriores.

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Áreas para Melhoria

- [ ] Suporte para números (0-9)
- [ ] Reconhecimento de palavras completas
- [ ] Interface gráfica (GUI)
- [ ] Modelo neural mais avançado
- [ ] Suporte para múltiplas mãos
- [ ] Testes automatizados
- [ ] CI/CD pipeline

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👨‍💻 Autor

**Gabryel Lima**
- LinkedIn: [Gabryel Lima](https://www.linkedin.com/in/gabryel-lima-9076541b2/)
- GitHub: [@Gabryel-lima](https://github.com/Gabryel-lima)

## 🙏 Agradecimentos

- **MediaPipe** pela biblioteca de detecção de mãos
- **OpenCV** pelo processamento de imagens
- **Scikit-learn** pelos algoritmos de ML
- **Comunidade Python** pelo suporte

## 📞 Suporte

Se você encontrar algum problema ou tiver dúvidas:

- 📧 Email: gabbryellimasi@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/Gabryel-lima/LibrIA/issues)
- 💬 Discussões: [GitHub Discussions](https://github.com/Gabryel-lima/LibrIA/discussions)

---

<div align="center">

⭐ **Se este projeto te ajudou, considere dar uma estrela!** ⭐

</div>
