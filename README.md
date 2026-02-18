# LibrIA - Reconhecimento de Libras com Visão Computacional

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18+-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

</div>

## 📖 Sobre o Projeto

O **LibrIA** é um sistema completo de reconhecimento de linguagem de sinais (Libras) utilizando técnicas de visão computacional e machine learning com modelo Random Forest. O projeto implementa todo o pipeline, desde a coleta de dados até a inferência em tempo real, desenvolvido do zero.

### 🎯 Objetivos

- **Acessibilidade**: Facilitar a comunicação entre pessoas surdas e ouvintes
- **Educação**: Apoiar o aprendizado de Libras
- **Tecnologia**: Demonstrar aplicações práticas de IA em visão computacional
- **Inovação**: Criar soluções inclusivas usando machine learning

### ✨ Características

- 🎥 **Captura em tempo real** via webcam
- 🤖 **Modelo de IA** com 99% de acurácia
- 📊 **Pipeline completo** de dados
- 🔄 **Inferência contínua** com feedback visual
- 📱 **Interface intuitiva** com overlay de informações
- 🏗️ **Arquitetura modular** e bem organizada

## 📚 Documentação Completa

**👉 [Índice Completo de Documentação](DOCUMENTATION_INDEX.md)** - Referência rápida para encontrar o que você precisa

### 🚀 Comece Aqui
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Como contribuir para o projeto
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** - Setup de desenvolvimento local
- **[docs/PULL_REQUEST_GUIDE.md](docs/PULL_REQUEST_GUIDE.md)** - Guia de Pull Requests

### 📖 Documentação do Projeto
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
## �🛠️ Tecnologias Utilizadas

| Tecnologia | Versão | Propósito |
|------------|--------|-----------|
| **Python** | 3.11+ | Linguagem principal |
| **TensorFlow** | 2.18.0 | Deep Learning (opcional) |
| **Keras** | 3.8.0 | API de alto nível |
| **PyTorch** | 2.5.1 | Alternativa de Deep Learning |
| **OpenCV** | 4.11+ | Captura e processamento de vídeo |
| **MediaPipe** | 0.10+ | Detecção de landmarks da mão |
| **Scikit-learn** | 1.6+ | Random Forest (modelo principal) |
| **NumPy** | 2.0+ | Computação numérica |
| **Pandas** | 2.2+ | Manipulação de dados |
| **Matplotlib** | 3.10+ | Visualização |
| **Serialização** | pickle | Salvar modelos e dados |

## 📁 Nova Estrutura do Projeto

```
LibrIA/
├── 📁 src/                          # Código fonte principal
│   ├── 📁 data_collection/          # Coleta de dados via webcam
│   │   ├── __init__.py
│   │   └── libras_data_collector.py
│   ├── 📁 data_processing/          # Processamento de imagens
│   │   ├── __init__.py
│   │   └── libras_dataset_processor.py
│   ├── 📁 model_training/           # Treinamento de modelos
│   │   ├── __init__.py
│   │   └── libras_model_trainer.py
│   ├── 📁 inference/                # Inferência em tempo real
│   │   ├── __init__.py
│   │   └── libras_realtime_classifier.py
│   └── __init__.py
├── 📁 config/                       # Configurações centralizadas
│   ├── __init__.py
│   └── settings.py
├── 📁 utils/                        # Utilitários e funções auxiliares
│   ├── __init__.py
│   └── helpers.py
├── 📁 data/                         # Dataset de imagens coletadas
│   ├── 0/                          # Classe A (0-299 imagens)
│   ├── 1/                          # Classe B (0-299 imagens)
│   └── ...                         # Outras classes (2-25)
├── 📁 dataset/                      # Dados processados
│   └── data.pickle                 # Dataset com landmarks
├── 📁 model/                        # Modelos treinados
│   └── model.pickle                # Modelo Random Forest
├── 📁 output/                       # Saídas (vídeos, screenshots)
├── 📁 backup_old_files/             # Arquivos antigos (backup)
├── � .deprecated-files/            # Files: Pipfile, setup scripts (obsoletos)
├── 📄 Makefile                      # Automação de tarefas (★ NOVO!)
├── 📄 main.py                       # Script principal unificado
├── 📄 collect_j_z.py               # Script específico para J e Z
├── 📄 requirements.txt              # Dependências principais
├── 📄 requirements-dev.txt          # Dependências de desenvolvimento
├── 📄 requirements-gpu.txt          # Suporte GPU (CUDA 12.4)
├── 📄 requirements.lock             # Lock file (versões exatas)
├── 📄 README.md                     # Documentação
├── 📄 test_setup.py                 # Testes de instalação
├── 📄 .gitignore                    # Arquivos ignorados pelo Git
└── 📄 .env.example                  # Variáveis de ambiente (exemplo)
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
make collect              # Coleta dados com webcam (A-Z)
make collect-jz           # Coleta apenas J e Z
make process              # Processa dataset (extrai landmarks)
make train                # Treina modelo Random Forest
make infer                # Inferência em tempo real
make run                  # Executa pipeline completo (collect→process→train→infer)

# Desenvolvimento
make test                 # Executa testes
make lint                 # Verifica código (requer dev dependencies)
make format               # Formata código (requer dev dependencies)
make install-dev          # Instala dependências de desenvolvimento

# Utilitários
make clean                # Remove __pycache__, arquivos temporários
make clean-all            # Remove tudo (venv, dados, modelos) - CUIDADO!
make freeze               # Gera requirements.lock com versões exatas
make update               # Atualiza dependências
make status               # Alias para 'environment'
```

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
python main.py collect     # Coletar dados
python main.py process     # Processar dataset
python main.py train       # Treinar modelo
python main.py infer       # Inferência em tempo real
```

### 🔄 Pipeline Completo

Para executar todo o pipeline de uma vez:

```bash
python main.py all
```

Este comando irá:
1. ✅ Coletar dados via webcam
2. ✅ Processar imagens e extrair landmarks
3. ✅ Treinar modelo Random Forest
4. ✅ Executar inferência em tempo real

### 📝 Instruções Detalhadas

#### 1. Coleta de Dados (`collect`)

```bash
python main.py collect
```

**Instruções durante a coleta:**
- O sistema irá percorrer todas as letras do alfabeto (A-Y, excluindo J e Z)
- Para cada letra, você deve fazer o sinal correspondente
- Pressione **'m'** para iniciar a captura de cada mão
- O sistema captura 150 imagens por mão (300 total por letra)
- Pressione **'q'** para sair a qualquer momento

#### 1.1. Coleta Específica de J e Z (`collect_jz`)

```bash
python main.py collect_jz
# ou diretamente:
python collect_j_z.py
```

**Quando usar:**
- Quando você já tem dados de A-Y e precisa apenas completar com J e Z
- Para adicionar as letras faltantes ao dataset existente

**Instruções específicas:**
- **Letra J**: Faça o sinal de J em Libras (mão em forma de gancho, movendo em círculo)
- **Letra Z**: Faça o sinal de Z em Libras (dedo indicador traçando a forma de Z)
- Pressione **'m'** para iniciar a captura de cada mão
- Serão capturadas 150 imagens por mão (300 total por letra)
- Pressione **'q'** para sair a qualquer momento

**Dicas importantes:**
- Certifique-se de fazer variações do sinal (ângulos diferentes)
- Mantenha boa iluminação para melhor detecção
- Centralize a mão na câmera durante a captura

#### 2. Processamento do Dataset (`process`)

```bash
python main.py process
```

Este script:
- Extrai landmarks das mãos usando MediaPipe
- Normaliza as coordenadas
- Salva o dataset processado em `dataset/data.pickle`

#### 3. Treinamento do Modelo (`train`)

```bash
python main.py train
```

**Resultados esperados:**
- Acurácia: ~99%
- Modelo salvo em `model/model.pickle`

#### 4. Inferência em Tempo Real (`infer`)

```bash
python main.py infer
```

**Controles:**
- **'q'**: Sair do programa
- **'r'**: Alternar gravação de vídeo
- **'s'**: Capturar screenshot
- **Detecção automática**: A cada 20 frames
- **Feedback visual**: Retângulo verde + letra prevista

## 📥 Datasets e Downloads

### Opção 1: Coletar seus próprios dados
Se você deseja coletar seus próprios dados de Libras, basta executar:
```bash
python main.py collect
```

### Opção 2: Usar datasets pré-coletados

#### ASL Alphabet Dataset (Kaggle)
Para usar o dataset ASL Alphabet Dataset (alfabeto em linguagem de sinais americana):
- 📊 **Dataset**: [ASL Alphabet Dataset - Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Estrutura**: 87.000 imagens das 26 letras
- **Instruções de download**:
  1. Crie conta no [Kaggle](https://www.kaggle.com)
  2. Baixe o dataset
  3. Descompacte em `data/archive/ASL_Alphabet_Dataset/`
  4. Execute `python main.py process`

#### Libras Alphabet Dataset (Coletado localmente)
Se você já coletou dados de Libras e quer compartilhar:
- 📁 **Pasta de dados**: `ASL_Alphabet_Dataset/`
- Para usar: Copie os dados coletados para `data/`

### Opção 3: Modelos pré-treinados
Se deseja usar modelos já treinados sem treinar do zero:
- 🤖 **Random Forest Model**: 
  - Localização: `model/model.pickle`
  - Para usar: `python main.py infer`

### Opção 4: Baixar dados processados (ZIP)
Para desenvolvimento rápido, você pode baixar os dados já processados em formato ZIP.

**Recursos com links de download** (quando disponíveis):
- 💾 **Google Drive**: Para compartilhamento fácil de arquivos
- ☁️ **Amazon S3**: Para distribuição em larga escala  
- 📦 **GitHub Releases**: Para pequenos arquivos (< 100MB)

Para mais detalhes, consulte a [documentação de datasets](docs/DATASETS.md).

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

**Nota**: O sistema agora inclui suporte completo para todas as 26 letras do alfabeto. Use `python collect_j_z.py` para capturar especificamente as letras J e Z se elas estiverem faltando no seu dataset.

## 🔬 Arquitetura Técnica

### Pipeline de Dados

1. **Captura**: Webcam → OpenCV
2. **Detecção**: MediaPipe → Landmarks da mão
3. **Processamento**: Normalização de coordenadas
4. **Treinamento**: Random Forest Classifier
5. **Inferência**: Classificação em tempo real

### Características do Modelo Random Forest

- **Algoritmo**: Random Forest Classifier
- **Features**: 42 coordenadas normalizadas (21 landmarks × 2 coordenadas)
- **Acurácia**: 99%
- **Tempo de resposta**: ~50ms por frame
- **Número de estimadores**: 100 (padrão)

### 🚀 Modelos Alternativos em Desenvolvimento

O projeto também possui uma implementação de **Transformer-based models** para reconhecimento mais avançado. Para explorar modelos experimenrais e futuras arquiteturas de deep learning, confira:

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
# Extração de landmarks
results = hands.process(img_rgb)
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Normalização das coordenadas
        for landmark in hand_landmarks.landmark:
            x = landmark.x - min(x_coords)
            y = landmark.y - min(y_coords)
```

## 📈 Resultados e Performance

### Métricas de Avaliação

- **Acurácia**: 99%
- **Precisão**: 98.5%
- **Recall**: 99.2%
- **F1-Score**: 98.8%

### Performance em Tempo Real

- **FPS**: ~20-30 frames/segundo
- **Latência**: <50ms
- **Uso de CPU**: ~15-25%
- **Uso de RAM**: ~200-300MB

## 🎥 Demonstração

[![Vídeo de Demonstração](https://img.shields.io/badge/YouTube-Demonstração-red)](https://www.linkedin.com/posts/joao-emanuel-7bb2981a4_projeto-de-vis%C3%A3o-computacional-com-linguagem-activity-7345904031329845248-s-o2?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC-9a38B_ih9uTXawvKzjklse66Jn0wYGio)

## 🔧 Personalização

### Ajustando Parâmetros

Você pode modificar os parâmetros no arquivo `config/settings.py`:

```python
# Configurações de coleta
DATASET_SIZE = 150          # Imagens por classe

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

1. Modifique `ALPHABET_DICT` em `config/settings.py`
2. Execute `python main.py collect` para as novas classes
3. Reprocesse com `python main.py process`
4. Retreine com `python main.py train`

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

Os arquivos de código antigos ainda estão em `backup_old_files/` para referência:
- `collect_data.py` → `src/data_collection/libras_data_collector.py`
- `create_dataset.py` → `src/data_processing/libras_dataset_processor.py`
- `model.py` → `src/model_training/libras_model_trainer.py`
- `inference_classifier.py` → `src/inference/libras_realtime_classifier.py`

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
