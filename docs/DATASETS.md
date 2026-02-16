# 📊 Datasets e Modelos - LibrIA

Este documento descreve todas as opções disponíveis para obter datasets e modelos pré-treinados para o projeto LibrIA.

## 📥 Opções de Datasets

### 1. Coletar Seus Próprios Dados (Recomendado para Libras)

**Quando usar**: Para coletar dados de Libras com sua própria câmera.

```bash
# Coletar alfabeto completo (A-Z)
python main.py collect

# Coletar apenas J e Z (complementar dataset)
python main.py collect_jz
```

**Características**:
- ✅ Dados frescos e personalizados
- ✅ Sincronizado com seu ambiente de iluminação
- ✅ Possibilidade de coletar variações
- ⏱️ Tempo: ~2-3 horas para alfabeto completo

**Estrutura gerada**:
```
data/
├── 0/  (Letra A)
├── 1/  (Letra B)
├── ... (Letras C-Y)
└── 25/ (Letra Z)
```

---

### 2. ASL Alphabet Dataset (Kaggle)

**Dataset Público**: [ASL Alphabet - Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

**Informações**:
- 📊 **Total**: 87.000+ imagens
- 🎯 **Classes**: 26 letras do alfabeto
- 📐 **Resolução**: Variada (150x150 a 300x300)
- 👥 **Diversidade**: Múltiplas pessoas, ângulos, iluminações

**Como baixar**:

#### Opção A: Via Kaggle CLI
```bash
# 1. Instale kaggle CLI
pip install kaggle

# 2. Baixe suas credenciais de API em https://www.kaggle.com/account
# 3. Coloque o arquivo em ~/.kaggle/kaggle.json

# 4. Baixe o dataset
kaggle datasets download -d grassknoted/asl-alphabet
unzip asl-alphabet.zip -d data/archive/
```

#### Opção B: Download manual
1. Visite [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
2. Clique em "Download"
3. Descompacte em `data/archive/ASL_Alphabet_Dataset/`

**Estrutura esperada**:
```
data/archive/ASL_Alphabet_Dataset/
├── asl_alphabet_train/
│   ├── A/
│   ├── B/
│   ├── ...
│   └── Z/
└── asl_alphabet_test/
    ├── A/
    ├── B/
    ├── ...
    └── Z/
```

**Como usar**:
```bash
# Processar o dataset
python main.py process

# Treinar modelo
python main.py train
```

---

### 3. Libras Alphabet Dataset (Local)

**Localização**: `ASL_Alphabet_Dataset/`

Se você já coletou dados de Libras:

```
ASL_Alphabet_Dataset/
├── asl_alphabet_train/
│   ├── A/ (imagens de A)
│   ├── B/ (imagens de B)
│   └── ...
└── asl_alphabet_test/
    ├── A/
    ├── B/
    └── ...
```

---

## 🤖 Modelos Pré-Treinados

### Random Forest Model

**Localização**: `model/model.pickle`

**Características**:
- ✅ Acurácia: 99%
- ✅ Tempo de inferência: ~50ms/frame
- ✅ Leve: ~2MB
- ✅ Sem dependências GPU

**Como usar**:
```bash
python main.py infer
```

### Modelos Deep Learning (Experimental)

**Localização**: `model/`

Modelos adicionais disponíveis:
- `best_temporal_model.h5` - Modelo temporal com CNN
- `temporal_cnn_model.h5` - CNN para processamento temporal
- `asl_vgg16_best_weights.keras` - VGG16 pré-treinado (src/saved/)

---

## 📦 Zips e Arquivos Compactados

### Dados Processados (data.pickle)

Já com landmarks extraídos e prontos para treino:

**Localização**: `dataset/data.pickle`

**Como obter**:
```bash
# Opção 1: Processar dados você mesmo
python main.py process

# Opção 2: Baixar arquivo pré-processado (quando disponível)
# [Link do Google Drive ou S3]
```

**Tamanho esperado**: ~50-100MB

---

## 🌐 Compartilhamento de Arquivos

### Para distribuir seus ZIPs:

#### Opção 1: Google Drive
```
1. Crie uma pasta no Google Drive
2. Upload dos zips
3. Compartilhe com permissão de leitura
4. Copie o link de compartilhamento
```

**Exemplo de link**:
```
https://drive.google.com/file/d/FILE_ID/view?usp=sharing
```

#### Opção 2: GitHub Releases
```bash
# Crie um release com os arquivos
git tag v1.0-dataset
git push origin v1.0-dataset
# Upload manual dos arquivos no GitHub
```

#### Opção 3: AWS S3 ou Google Cloud Storage
```bash
# Ideal para grandes volumes de dados
# Consulte a documentação dos serviços
```

---

## 📋 Checklist de Setup

- [ ] Escolhi uma fonte de dados
- [ ] Download dos dados iniciado
- [ ] Dados descompactados na pasta correta
- [ ] Executei `python main.py process`
- [ ] Modelo foi treinado com `python main.py train`
- [ ] Teste de inferência executado com `python main.py infer`

---

## 🔗 Links Importantes

| Recurso | Link |
|---------|------|
| **Kaggle** | https://www.kaggle.com/datasets/grassknoted/asl-alphabet |
| **GitHub do Projeto** | https://github.com/Gabryel-lima/LibrIA |
| **MediaPipe** | https://mediapipe.dev |
| **OpenCV** | https://opencv.org |

---

## ❓ Dúvidas Frequentes

### P: Posso usar o ASL Dataset para Libras?
**R**: Sim, pois ambos são linguagens de sinais com gestos semelhantes. Use-o para pré-treinamento e ajuste fino com dados de Libras.

### P: Qual é o tamanho total dos dados?
**R**: 
- ASL Dataset: ~2GB
- Dados coletados manualmente: ~500MB-1GB
- Dataset processado (pickle): ~100MB

### P: Posso combinar múltiplos datasets?
**R**: Sim! Coloque todas as imagens em `data/` e processe juntas com `python main.py process`.

### P: O modelo funciona sem GPU?
**R**: Sim! O Random Forest e MediaPipe funcionam bem em CPU. Para deep learning, recomenda-se GPU, mas CPU ainda é funcional.

---

**Última atualização**: 15 de fevereiro de 2026
