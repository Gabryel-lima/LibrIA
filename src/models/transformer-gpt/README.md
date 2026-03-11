# Diretório `src/models/transformer-gpt`

Este diretório contém uma prévia de um modelo baseado na arquitetura **Transformer GPT**. O modelo aqui presente serve como uma base inicial para estudos e experimentações de arquiteturas de deep learning avançadas.

## 📚 Sobre a Arquitetura Transformer-GPT

Ainda é necessário aprofundar o estudo sobre a arquitetura Transformer GPT para compreender melhor seu funcionamento, suas aplicações e como otimizá-la para diferentes casos de uso. Recomenda-se explorar a documentação oficial e materiais de referência sobre a arquitetura GPT para obter um entendimento mais sólido.

## 🎯 Dados para Treinamento

### Opção 1: Dados do pipeline atual do LibrIA

#### Dataset estático unificado

O projeto gera amostras estáticas em:

```
dataset/static/<label>/sample_XXX.npy
```

Cada arquivo contém landmarks normalizados por amostra. No modo padrão `wrist_relative`, o shape salvo é `(21, 3)`.

**Dimensionalidade**:

- `bounding_box`: 42 features
- `wrist_relative`: 63 features

O padrão atual do projeto é `wrist_relative`, então a expectativa principal para novos experimentos é trabalhar com 63 features por frame.

#### Dataset temporal recomendado

Para modelos sequenciais, o formato mais alinhado ao código atual está em:

```
dataset/temporal/<label>/seq_XXX.npy
```

**Formato dos dados:**
- **Input**: sequências de landmarks por frame
- **Shape esperado no padrão atual**: `(num_amostras, 30, 21, 3)`
- **Labels**: classes como `J`, `Z` ou outras classes coletadas

**Como preparar:**
```bash
# 1. Coleta temporal
make collect-temporal SEQUENCE_LABELS=J\ Z SEQUENCE_COUNT=30 SEQUENCE_LENGTH=30

# 2. Se necessário, adapte os .npy para o formato consumido pelo experimento Transformer
# 3. Use o mesmo FEATURE_MODE do restante do projeto para evitar incompatibilidade
```

#### Imagens Brutas
Também é possível treinar diretamente com imagens:

```
ASL_Alphabet_Dataset/
├── asl_alphabet_train/  # 26 pastas (A-Z)
├── asl_alphabet_test/   # Dataset de teste
```

**Formato:**
- **Input**: Imagens RGB (640×480 ou redimensionadas)
- **Output**: Landmarks extraídos com MediaPipe
- **Total**: ~87.000 imagens para ASL Alphabet Dataset

### Opção 2: Dados Externos

#### 1. **Vídeos de Libras/ASL**
- Extrair frames dos vídeos
- Processar com MediaPipe para obter landmarks
- Agrupar em sequências temporais

#### 2. **Datasets Públicos**
- **ASL Alphabet Dataset** (Kaggle): 87.000 imagens
- **WLASL Dataset**: 21.000 vídeos de linguagem de sinais americana
- **LSF Dataset**: 5.000+ sinais em Língua de Sinais Francesa
- **DGS Dataset**: Dados em Língua de Sinais Alemã

#### 3. **Dados Sintéticos**
- Gerar dados augmentados a partir dos existentes
- Usar data augmentation (rotação, zoom, perspectiva)
- Criar variações de ângulos e posições

### Opção 3: Seu Próprio Dataset

Você pode criar um dataset customizado:

```
seu_dataset/
├── letra_A/
│   ├── video_1.mp4
│   ├── video_2.mp4
│   └── ...
├── letra_B/
│   └── ...
└── ...
```

**Processamento:**
```python
# Extrair landmarks com o mesmo formato do projeto
# Use scripts/collect_dataset.py como referência para gravar amostras em NPY
```

## 🚀 Uso Independente

### Instalação de Dependências

```bash
# Instale as dependências necessárias
pip install tensorflow keras torch numpy opencv-python mediapipe

# Ou use o requirements do projeto
pip install -r requirements.txt
pip install transformer  # Dependência adicional para modelos Transformer
```

### Treinamento Standalone

```python
# Exemplo de uso independente
import numpy as np
from src.models.transformer_gpt import TransformerGPTModel

# 1. Carregar dados
landmarks_data = np.load('dataset/landmarks.npy')  # Ex.: (n_samples, n_frames, 63)
labels = np.load('dataset/labels.npy')              # Shape: (n_samples,)

# 2. Criar e configurar modelo
model = TransformerGPTModel(
    input_dim=63,           # Padrão atual: wrist_relative
    sequence_length=20,     # Número de frames por sequência
    num_classes=26,         # A-Z
    d_model=256,            # Dimensão interna
    nhead=8,                # Número de attention heads
    num_layers=6,           # Número de blocos Transformer
)

# 3. Compilar
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Treinar
history = model.fit(
    landmarks_data, 
    labels,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# 5. Salvar modelo
model.save('transformer_gpt_model.h5')
```

### Inferência Standalone

```python
from src.models.transformer_gpt import TransformerGPTModel

# Carregar modelo treinado
model = TransformerGPTModel.load('transformer_gpt_model.h5')

# Fazer predição
novo_landmarks = extrair_landmarks_do_video('novo_video.mp4')
predicao = model.predict(novo_landmarks)
classe_predita = np.argmax(predicao)

print(f"Letra predita: {ALPHABET[classe_predita]}")
```

## 🔗 Integração com Projeto LibrIA

### Opção 1: Substituir Random Forest por Transformer-GPT

```bash
# 1. Treinar modelo Transformer com dados do projeto
python train_transformer.py

# 2. Modificar config/settings.py
MODEL_TYPE = 'transformer'  # ao invés de 'random_forest'

# 3. Usar na inferência
python main.py infer
```

### Opção 2: Usar como Ensemble

Combinar Random Forest + Transformer para melhor acurácia:

```python
# Em src/inference/libras_realtime_classifier.py

class EnsembleClassifier:
    def __init__(self):
        self.rf_model = load_random_forest()
        self.transformer_model = load_transformer()
    
    def predict(self, landmarks):
        # Predição Random Forest
        pred_rf = self.rf_model.predict(landmarks)
        
        # Predição Transformer
        pred_transformer = self.transformer_model.predict(landmarks)
        
        # Combinar predições (votação ou média)
        final_pred = (pred_rf + pred_transformer) / 2
        return np.argmax(final_pred)
```

### Opção 3: Usar para Reconhecimento Temporal de Sequências

O Transformer é excelente para sequências. Use para reconhecer:

```python
# Reconhecer uma sequência de gestos (palavras em Libras)
class SequenceRecognizer:
    def __init__(self, transformer_model):
        self.model = transformer_model
    
    def recognize_gesture_sequence(self, frames):
        """
        Reconhecer uma sequência de 20-50 frames
        que pode representar uma palavra completa em Libras
        """
        landmarks_sequence = self.extract_landmarks(frames)
        prediction = self.model.predict(landmarks_sequence)
        return prediction
```

## 📊 Comparação: Random Forest vs Transformer-GPT

| Aspecto | Random Forest | Transformer-GPT |
|---------|---|---|
| **Dados necessários** | ~300-500 imagens por classe | 1000+ imagens ou sequências |
| **Tempo de treinamento** | Minutos | Horas |
| **Acurácia estática** | 99% | ~95-98% (inicialmente) |
| **Reconhecimento temporal** | Não | ✅ Excelente |
| **Reconhecimento de sequências** | Limitado | ✅ Muito bom |
| **Interpretabilidade** | Alta | Baixa |
| **Requisitos computacionais** | Baixos | Altos (GPU recomendado) |

## 🎯 Próximas Passos

1. **Implementar modelo completo** em `transformer_gpt.py`
2. **Criar script de treinamento** (`train_transformer.py`)
3. **Integrar com pipeline** do LibrIA
4. **Testar em tempo real** com inferência
5. **Comparar resultados** com Random Forest
6. **Otimizar arquitetura** para Libras
7. **Documentar hyperparâmetros** ideais

## 📚 Recursos Recomendados

- **"Attention Is All You Need"** - Vaswani et al. (Artigo original Transformer)
- **HuggingFace Transformers**: https://huggingface.co/transformers/
- **PyTorch Transformer**: https://pytorch.org/docs/stable/nn.html#transformer-layers
- **TensorFlow Transformer**: https://www.tensorflow.org/text/guide/transformer

## 🔗 Links Úteis

- [Documentação MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)
- [WLASL Dataset](https://github.com/dxli94/WLASL)
- [Transformers.js](https://github.com/xenova/transformers.js) - Para usar em browser
