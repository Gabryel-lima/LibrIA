# 🤝 Guia de Contribuição - LibrIA

Obrigado por considerar contribuir para o **LibrIA**! Este documento fornece diretrizes e instruções para colaboradores.

## 📋 Índice

- [Código de Conduta](#código-de-conduta)
- [Como Contribuir](#como-contribuir)
- [Processo de Pull Request](#processo-de-pull-request)
- [Padrões de Código](#padrões-de-código)
- [Testes](#testes)
- [Documentação](#documentação)
- [Reportando Bugs](#reportando-bugs)
- [Sugerindo Melhorias](#sugerindo-melhorias)

## 🤝 Código de Conduta

Todos os contribuidores devem seguir nosso [Código de Conduta](CODE_OF_CONDUCT.md). Resumidamente:
- Seja respeitoso e inclusivo
- Aceite críticas construtivas
- Foque no que é melhor para a comunidade
- Mostre empatia com outros colaboradores

## 💡 Como Contribuir

### Não sabe por onde começar?

1. **Issues com `good-first-issue`** - Perfeito para novatos
2. **Issues com `help-wanted`** - Precisamos de ajuda nessas áreas
3. **Melhorias de documentação** - Sempre bem-vindas!
4. **Correção de bugs** - Veja issues abertas

### Áreas de Contribuição

| Área | Descrição | Nível |
|------|-----------|-------|
| **Coleta de Dados** | Ampliar dataset de gestos | Iniciante-Intermediário |
| **Modelo ML** | Melhorar acurácia e performance | Avançado |
| **Documentação** | Traduzir, melhorar guias | Iniciante |
| **Testes** | Aumentar cobertura de testes | Intermediário |
| **Otimização** | Melhorar velocidade/latência | Avançado |
| **UX/UI** | Melhorar interfaces | Intermediário |
| **Acessibilidade** | Tornar mais inclusivo | Intermediário |

## 🔄 Processo de Pull Request

### 1. Preparação

```bash
# Fork o repositório
# Clone seu fork
git clone https://github.com/Gabryel-lima/LibrIA.git
cd LibrIA

# Crie uma branch para sua feature/fix
git checkout -b feat/descriptive-name
# ou
git checkout -b fix/bug-description
```

### 2. Branch Naming Convention

Use os seguintes prefixos:

- **`feat/`** - Nova funcionalidade: `feat/improved-hand-detection`
- **`fix/`** - Correção de bug: `fix/model-inference-crash`
- **`docs/`** - Documentação: `docs/contributing-guide`
- **`test/`** - Testes: `test/add-unit-tests`
- **`refactor/`** - Refatoração: `refactor/simplify-data-pipeline`
- **`perf/`** - Performance: `perf/optimize-inference-speed`

### 3. Faça suas Alterações

```bash
# Configure o ambiente de desenvolvimento
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instale dependências de desenvolvimento
pip install -r requirements-dev.txt

# Faça suas alterações
# ... edite os arquivos ...

# Teste suas mudanças
python -m pytest tests/ -v
```

### 4. Commit Message Convention

Seguimos [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Exemplos:**

```
feat(inference): add support for face recognition

Implements face recognition alongside hand gesture recognition
to improve overall accuracy in mixed hand-face scenarios.

Closes #123
```

```
fix(data-collector): prevent duplicate frame capture

Solves race condition in frame buffering that caused
duplicate frames in dataset collection.

Closes #456
```

**Types:**
- `feat`: Nova funcionalidade
- `fix`: Correção de bug
- `docs`: Mudanças na documentação
- `style`: Formatação, sem lógica (whitespace, semicolons, etc)
- `refactor`: Refatoração sem mudança de funcionalidade
- `perf`: Mudanças que melhoram performance
- `test`: Adicionar ou atualizar testes
- `ci`: Mudanças em CI/CD

### 5. Envie o Pull Request

```bash
# Push sua branch
git push origin feat/descriptive-name

# Abra um PR no GitHub
```

#### Checklist do PR

Certifique-se de que seu PR inclui:

- [ ] Descrição clara do que foi mudado e por quê
- [ ] Referência a issues relacionadas (`Closes #123`)
- [ ] Testes novos/atualizados (se aplicável)
- [ ] Documentação atualizada
- [ ] Sem conflitos com `develop`
- [ ] Código segue os padrões do projeto
- [ ] Commits estão bem organizados

#### Template do PR

```markdown
## 📝 Descrição

Descreva as mudanças realizadas.

## 🎯 Tipo de Mudança

- [ ] 🐛 Bug fix
- [ ] ✨ Nova funcionalidade
- [ ] 📚 Documentação
- [ ] ♻️ Refatoração
- [ ] 🚀 Performance

## 🔗 Issues Relacionadas

Closes #123

## 🧪 Como Testar?

Descreva os passos para testar as mudanças.

## 📸 Screenshots (se aplicável)

Adicione screenshots para mudanças na UI.

## ✅ Checklist

- [ ] Meu código segue os padrões do projeto
- [ ] Executei testes localmente com sucesso
- [ ] Adicionei testes para novas funcionalidades
- [ ] Atualizei a documentação relevante
- [ ] Sem mudanças que quebrem backward compatibility
```

## 📐 Padrões de Código

### Python

```python
# ✅ BOM
def process_hand_landmarks(landmarks: List[Tuple[float, float]]) -> np.ndarray:
    """
    Process hand landmarks for model inference.
    
    Args:
        landmarks: List of (x, y) coordinate tuples
        
    Returns:
        Processed landmarks as numpy array
    """
    return np.array(landmarks).flatten()


# ❌ RUIM
def process(l):
    return np.array(l).flatten()
```

### Estilo de Código

```bash
# Use Black para formatação
black --line-length 100 src/

# Use isort para organizar imports
isort src/

# Use Pylint para linting
pylint src/ --disable=too-few-public-methods
```

### Type Hints (PEP 484)

Sempre use type hints em funções públicas:

```python
from typing import List, Tuple, Optional

def extract_features(
    image: np.ndarray,
    detector: HandDetector,
    filter_empty: bool = True
) -> Optional[np.ndarray]:
    """Extract hand features from image."""
    ...
```

### Docstrings (NumPy Style)

```python
def train_model(data: np.ndarray, labels: np.ndarray) -> RandomForestClassifier:
    """
    Train a Random Forest model on hand gesture data.
    
    Parameters
    ----------
    data : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    labels : np.ndarray
        Label vector of shape (n_samples,)
        
    Returns
    -------
    RandomForestClassifier
        Trained model ready for inference
        
    Raises
    ------
    ValueError
        If data shape doesn't match expected dimensions
        
    Examples
    --------
    >>> data = np.random.rand(100, 84)
    >>> labels = np.random.randint(0, 26, 100)
    >>> model = train_model(data, labels)
    """
    ...
```

### Comentários

```python
# ✅ BOM - Explica o POR QUÊ
# Usar MediaPipe com GPU quando disponível para reduzir latência
# em mais de 50% em vídeo em tempo real
detector = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.7
)

# ❌ RUIM - Explica o QUÊ (óbvio do código)
# Criar detector de mãos
detector = mp.solutions.hands.Hands()
```

## 🧪 Testes

### Estrutura de Testes

```
tests/
├── __init__.py
├── test_data_collection.py
├── test_data_processing.py
├── test_model_training.py
└── fixtures/  # Dados para testes
```

### Executar Testes

```bash
# Todos os testes
pytest tests/ -v

# Testes com cobertura
pytest tests/ --cov=src --cov-report=html

# Teste específico
pytest tests/test_data_processing.py::test_normalize_landmarks -v

# Testes com markers
pytest -m "not slow" -v  # Pula testes lentos
```

### Exemplo de Teste

```python
import pytest
import numpy as np
from src.data_processing.libras_dataset_processor import LirasProcessor

class TestDataProcessor:
    """Test suite for data processing module."""
    
    @pytest.fixture
    def processor(self):
        """Factory fixture for processor."""
        return LibrasProcessor()
    
    def test_normalize_landmarks(self, processor):
        """Test landmark normalization."""
        # Arrange
        landmarks = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        # Act
        result = processor.normalize(landmarks)
        
        # Assert
        assert result.shape == landmarks.shape
        assert np.allclose(result.mean(), 0, atol=1e-7)
        
    def test_invalid_input(self, processor):
        """Test handling of invalid input."""
        with pytest.raises(ValueError):
            processor.normalize(np.array([]))
```

### Coverage Mínimo

- Código novo deve ter cobertura ≥ 80%
- Funções críticas devem ter 100% de cobertura

## 📚 Documentação

### Adicionar Nova Funcionalidade

1. **Docstring** na função/classe
2. **README.md** - Seção de uso se relevante
3. **docs/** - Guia detalhado se complexo
4. **CHANGELOG.md** - Adicione entrada

### Exemplo

Se adicionar novo modelo:

```markdown
# docs/MODELS.md

## Novo Modelo: EfficientNet

### Overview
...

### Usage
python main.py --model efficientnet

### Performance
...
```

## 🐛 Reportando Bugs

### Antes de Reportar

1. Verifique se o bug já foi reportado
2. Teste com a versão mais recente
3. Procure a solução na seção de problemas do [README](README.md#solução-de-problemas)

### Ao Reportar

Use o template:

```markdown
## Descrição do Bug
Descrição clara do problema.

## Passos para Reproduzir
1. ...
2. ...
3. ...

## Resultado Esperado
...

## Resultado Atual
...

## Ambiente
- OS: Linux / Windows / macOS
- Python: 3.11
- TensorFlow/PyTorch: versão X
- Acesso GPU: Sim/Não

## Logs
```
Adicione logs relevantes
```
```

## 💡 Sugerindo Melhorias

Use o template:

```markdown
## Descrição da Melhoria
Descrição clara da melhoria sugerida.

## Motivação
Por que isso seria útil?

## Implementação Proposta
Como você implementaria isso?

## Exemplos
Exemplos de uso.
```

## 📋 Checklist para Revisores

Ao revisar um PR, verifique:

- [ ] Código funciona conforme descrito
- [ ] Testes passam e cobertura adequada
- [ ] Sem conflitos com desenvolvedimento principal
- [ ] Segue padrões do projeto
- [ ] Documentação está atualizada
- [ ] Sem problemas de segurança óbvios
- [ ] Performance é aceitável
- [ ] Commits são bem organizados

## 🎓 Recursos Úteis

- [Setup de Desenvolvimento](docs/DEVELOPMENT.md)
- [Estrutura do Código](docs/ARCHITECTURE.md)
- [Guia de Pull Requests](docs/PULL_REQUEST_GUIDE.md)
- [Fase 1 — vocabulário e avaliação](docs/FASE1_RECONHECIMENTO.md)
- [Fase 2 — pipeline temporal](docs/FASE2_TEMPORAL.md)

## 🎉 Recompensas

Contribuidores ativos recebem:

- Menção no [CONTRIBUTORS.md](CONTRIBUTORS.md)
- Reconhecimento em releases
- Acesso a canal privado do Discord (em breve)
- Prioridade em featured projects

## 📞 Dúvidas?

- 💬 [Discussions](https://github.com/Gabryel-lima/LibrIA/discussions)
- 📧 Email: gabbryellimasi@gmail.com

---

**Obrigado por contribuir para LibrIA! 🙏**

Cada contribuição, por menor que seja, ajuda a tornar Libras mais acessível.
