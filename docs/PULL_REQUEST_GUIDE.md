# 📋 Guia de Pull Requests - LibrIA

Instruções detalhadas para criar e submeter pull requests de qualidade ao LibrIA.

## 🎯 Antes de Começar

### Pré-requisitos

- ✅ Você fez um fork do repositório
- ✅ Você clonoulocalmente: `git clone https://github.com/Gabryel-Lima/LibrIA.git`
- ✅ Você deixou seu fork sincronizado com upstream
- ✅ Você leu [CONTRIBUTING.md](../CONTRIBUTING.md)

### Padrões Esperados

Seu PR será avaliado por:
- ✅ Funcionalidade (código faz o que promete)
- ✅ Qualidade (segue padrões do projeto)
- ✅ Testes (cobertura mínima 80%)
- ✅ Documentação (atualizada)

---

## 🔄 Processo Passo-a-Passo

### 1. Crie uma Feature Branch

```bash
# Sincronize com upstream
git fetch upstream
git checkout develop
git pull upstream develop

# Crie branch para sua feature
git checkout -b feat/nome-descritivo
# ou para fix
git checkout -b fix/nome-do-bug
```

**Convenção de Nomes:**
- `feat/` - Nova funcionalidade
- `fix/` - Correção de bug
- `docs/` - Documentação
- `test/` - Testes
- `refactor/` - Refatoração
- `perf/` - Performance

### 2. Faça Suas Alterações

```bash
# Edite os arquivos
# ...seu código...

# Teste localmente
python -m pytest tests/ -v

# Formate código
black src/ --line-length 100
isort src/

# Verifique qualidade
pylint src/ --disable=too-few-public-methods
mypy src/ --ignore-missing-imports
```

### 3. Commit com Mensagens Claras

```bash
# Use o formato: type(scope): description
git commit -m "feat(inference): add support for face recognition"

# Ou para múltiplos commits:
git commit -m "fix(data-collector): prevent duplicate frame capture

Solves race condition in frame buffering
Closes #123"
```

**Formato:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Tipos:**
- `feat:` Nova funcionalidade
- `fix:` Correção de bug
- `docs:` Documentação
- `test:` Testes
- `refactor:` Refatoração
- `perf:` Performance
- `style:` Formatação
- `ci:` CI/CD

**Exemplo Completo:**
```
feat(model-training): improve model accuracy with data augmentation

- Implement random rotation and zoom
- Add data normalization
- Improve validation metrics

Closes #456
Related to #789
```

### 4. Push para Seu Fork

```bash
git push origin feat/nome-descritivo
```

### 5. Abra o Pull Request

Vá para GitHub e clique "Compare & pull request"

**No título do PR:**
```
[TIPO] Descrição breve do que foi feito
```

**Exemplos:**
```
[FEATURE] Add face recognition support
[FIX] Fix memory leak in inference
[DOCS] Update API documentation
```

**NO CORPO DO PR:**

Use o template padrão. Se não aparecer, copie:

```markdown
## 📝 Descrição

Descreva as mudanças realizadas de forma clara. O que foi modificado e por quê?

## 🎯 Tipo de Mudança

- [ ] 🐛 Bug fix
- [ ] ✨ Nova funcionalidade
- [ ] 📚 Documentação
- [ ] ♻️ Refatoração
- [ ] 🚀 Performance

## 🔗 Issues Relacionadas

Closes #123

## 🧪 Como Testar?

Descreva os passos para validar suas mudanças:

1. Instale dependências: `pip install -r requirements-dev.txt`
2. Execute: `python main.py --feature new-feature`
3. Verifique: resultado em `output/`

## 📸 Screenshots (se aplicável)

[Adicione screenshots de UI changes]

## ✅ Checklist

- [ ] Meu código segue os padrões do projeto
- [ ] Executei `black` e `isort` localmente
- [ ] Executei `pylint` sem erros
- [ ] Adicionei/atualizei testes
- [ ] Meus testes passam: `pytest tests/ -v`
- [ ] Cobertura está 80%+
- [ ] Atualizei documentação relevante
- [ ] Sem mudanças que quebram backward compatibility
- [ ] Commits estão bem estruturados
```

---

## 🔍 O Que Reviewers Procuram

### ✅ Bom

```python
# Código limpo com type hints
def extract_landmarks(
    image: np.ndarray,
    hand_detector: MediapipeDetector
) -> Optional[np.ndarray]:
    """
    Extract hand landmarks from image.
    
    Args:
        image: Input frame as numpy array
        hand_detector: Initialized detector
        
    Returns:
        Normalized landmarks or None
    """
    landmarks = hand_detector.detect(image)
    return normalize_landmarks(landmarks) if landmarks else None

# Teste com cobertura
def test_landmark_extraction():
    image = load_test_image()
    detector = MediapipeDetector()
    result = extract_landmarks(image, detector)
    assert result is not None
    assert result.shape == (21, 2)  # 21 landmarks, (x,y)

# Docstring completo
# Commit message clara
# Referência a issues
```

### ❌ Ruim

```python
# Sem type hints
def extract_landmarks(image, detector):
    landmarks = detector.detect(image)
    return normalize(landmarks)

# Sem teste
# Sem docstring
# Commit: "fix bug"  ← Vago!
# Sem referência a issue
```

---

## 📊 Ciclo de Review

```
1. PR Submetido
   ↓
2. Checks Automáticos
   - CI/CD passa?
   - Coverage ok?
   ↓
3. Code Review
   - 1-2 reviewers avaliam
   - Comentários construtivos
   ↓
4. Ajustes
   - Você responde feedback
   - Faz novos commits
   - NÃO use force-push!
   ↓
5. Aprovação
   - Reviewers aprovam
   ↓
6. Merge
   - Branch é mergeada
   - Branch é deletada automaticamente
```

---

## ⏱️ Tempo de Review

| Tipo | Tempo Esperado |
|------|----------------|
| Bug fix simples | 24-48h |
| Documentação | 24-48h |
| Pequena feature | 2-3 dias |
| Grande feature | 3-7 dias |
| Review crítica | pode ser mais |

---

## 🎓 Boas Práticas

### ✅ Faça

- ✅ PRs pequenos e focados (100-500 linhas idealmente)
- ✅ Descrever claramente o que e por quê
- ✅ Testar localmente antes de enviar
- ✅ Responder feedback rapidamente
- ✅ Agradecer reviewers
- ✅ Atualizar branch antes de merge
- ✅ Ser receptivo a críticas
- ✅ Referenciar issues relacionadas

### ❌ Evite

- ❌ PRs gigantes (1000+ linhas)
- ❌ Vago: "fix stuff"
- ❌ Enviar sem testar
- ❌ Brigar com reviewers
- ❌ Ignorar feedback
- ❌ Force-push para atualizar
- ❌ Mudar tópico no meio do PR
- ❌ Commits desorganizados

---

## 🚀 Acelerando o Review

### 1. PR Bem Descrito

```markdown
BORA VÊR:
- ✅ Título claro
- ✅ Descrição detalhada
- ✅ Issue referenciada
- ✅ Como testar
- ✅ Screenshots (se UI)
```

### 2. Código Limpo

```bash
# Rodar ANTES de fazer PR
black src/ --line-length 100
isort src/
pylint src/ --exit-zero
mypy src/ --ignore-missing-imports
pytest tests/ --cov=src --cov-report=html
```

### 3. Pequeno é Melhor

- Preferência: PR com 100-300 linhas
- Aceitável: 300-500 linhas
- Discussão: 500-1000 linhas
- Separe em múltiplos PRs: 1000+ linhas

### 4. Foco em Um Tópico

❌ Ruim:
```
- Refatorar utils
- Adicionar nova feature
- Corrigir 3 bugs
- Atualizar documentação
```

✅ Bom:
```
- Adicionar detecção de múltiplas mãos (feature + testes + docs)
```

---

## 🛠️ Command Reference

### Git

```bash
# Sincronizar fork
git remote add upstream https://github.com/Gabryel-Lima/LibrIA.git
git fetch upstream
git rebase upstream/develop

# Atualizar branch durante review
git pull upstream develop
git rebase origin/seu-branch

# Limpar commits antes de merge
git rebase -i HEAD~3  # Reorganizar últimos 3 commits
```

### Testes

```bash
# Rodar testes
pytest tests/ -v

# Com cobertura
pytest tests/ --cov=src --cov-report=html

# Teste específico
python -m unittest tests.test_temporal_pipeline -v

# Testes rápidos
pytest -m "not slow" -v
```

### Qualidade

```bash
# Formatação
black --check src/
isort --check-only src/

# Linting
pylint src/ --exit-zero

# Type checking  
mypy src/ --ignore-missing-imports

# Performance
python -m cProfile -s cumulative main.py
```

---

## 📋 Checklist Final Antes de PRs

```
PRE-SUBMISSION
  [ ] Código formatado (black + isort)
  [ ] Sem erros pylint/mypy
  [ ] Testes passam
  [ ] Coverage 80%+
  [ ] Sem conflitos com develop
  [ ] Commits bem estruturados
  [ ] Docstrings completos
  [ ] Sem print/TODO/FIXME

PR SUBMISSION
  [ ] Título claro
  [ ] Descrição detalhada
  [ ] Issue referenciada
  [ ] Como testar explicado
  [ ] Screenshots (se UI)
  [ ] Checklist completo no PR

DURANTE REVIEW
  [ ] Respondo feedback
  [ ] Não faço force-push
  [ ] Agradeço reviewers
  [ ] Sou receptivo a críticas
```

---

## 🆘 Problemas Comuns

### "Meu PR tem merge conflicts"

```bash
# Atualize local
git fetch upstream
git rebase upstream/develop

# Resolva conflicts no editor
# Then:
git add .
git rebase --continue
git push origin sua-branch --force-with-lease
```

### "CI falhou"

Clique em "Details" para ver o erro:
- Testes falharam? → Corrija código + push
- Coverage baixo? → Adicione testes
- Linting error? → Rode `black` e `isort`

### "Reviewer pediu muitas mudanças"

Isso é normal! Significa que o reviewer se importa 🙏
- Faça os ajustes com calma
- Novo commit é ok (não force-push)
- Pergunte se tiver dúvida

### "Fiz force-push, perdi histórico"

Não é o fim do mundo, mas evite.  
Próxima vez use:
```bash
git push --force-with-lease  # Mais seguro
```

---

## 📚 Recursos

- [CONTRIBUTING.md](../CONTRIBUTING.md) - Diretrizes gerais
- [Conventional Commits](https://www.conventionalcommits.org/) - Formato de commits
- [Pytest Docs](https://docs.pytest.org/) - Framework de testes
- [Black Formatting](https://black.readthedocs.io/) - Formatador Python

---

## 💬 Perguntas?

- 📧 Issues com label `question`
- 💬 GitHub Discussions

---

**Obrigado por contribuir! Cada PR é uma melhoria para o projeto. 💙**

Última atualização: 2026-02-17
