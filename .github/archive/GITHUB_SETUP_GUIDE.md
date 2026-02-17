# 🚀 Guia de Configuração do Repositório GitHub

Documento com todas as recomendações e passos para configurar corretamente seu repositório no GitHub.

## 📋 Índice

1. [Pré-requisitos](#pré-requisitos)
2. [Configuração Básica](#configuração-básica)
3. [Proteção de Branches](#proteção-de-branches)
4. [Segurança](#segurança)
5. [Automação](#automação)
6. [Community Profile](#community-profile)
7. [Checklist Final](#checklist-final)

---

## 🎯 Pré-requisitos

- ✅ Repositório criado e vazio no GitHub
- ✅ Permissão de Admin no repositório
- ✅ Todos os arquivos de documentação criados
- ✅ GitHub Actions habilitado

---

## ⚙️ Configuração Básica

### 1. Settings → General

```
✓ Description: "Sistema completo de reconhecimento de gestos de Libras 
               com visão computacional e machine learning"

✓ Website: https://seu-site.com (opcional)

✓ Topics: 
  - libras
  - sign-language
  - computer-vision
  - machine-learning
  - python
  - gesture-recognition
  - accessibility
  - asl

✓ Visibility: PUBLIC

✓ Default branch: develop
  (importante: não é main!)

✓ Features:
  ☑ Discussions (para comunidade)
  ☐ Wikis (documentação em docs/)
  ☑ Issues
  ☑ Projects

✓ Pull Requests:
  - Permite auto-merge: NÃO
  - Sugggest updating branches: SIM
  - Allow auto-delete branches: SIM
  - Include merge commits: NÃO
  - Allow squash merging: SIM
  - Allow rebase merging: NÃO
```

### 2. Crie os Labels

Vá em Settings → Labels e crie:

```bash
# Tipo
type/bug          #d73a4a  (Vermelho)
type/feature       #a2eeef  (Ciano)
type/docs         #0075ca  (Azul)
type/test         #ff6d00  (Laranja)
type/refactor     #7057ff  (Roxo)

# Prioridade
priority/critical  #ff0000  (Vermelho escuro)
priority/high     #ff6d00  (Laranja)
priority/medium   #ffaa00  (Amarelo)
priority/low      #90ee90  (Verde claro)

# Status
status/blocked    #ff0000  (Vermelho)
status/in-progress #ffaa00  (Amarelo)
status/review     #0075ca  (Azul)
status/stale      #cccccc  (Cinza)

# Community
good-first-issue  #7057ff  (Roxo)
help-wanted      #ff6d00  (Laranja)
rfc              #7057ff  (Roxo)
```

---

## 🔒 Proteção de Branches

### Branch: `main` (Produção)

Vá em Settings → Branches → Add rule

```
Branch name pattern: main

✓ Require a pull request before merging
  ✓ Require approvals: 2 reviewers
  ✓ Require review from Code Owners: SIM
  ✓ Dismiss stale pull request approvals: SIM
  ✓ Require status checks to pass:
    - build
    - test
    - code-quality

✓ Require branches to be up to date before merging
✓ Include administrators in restrictions
✓ Restrict who can push to matching branches:
  Only Maintainers
```

### Branch: `develop` (Desenvolvimento)

```
Branch name pattern: develop

✓ Require a pull request before merging
  ✓ Require approvals: 1 reviewer
  ✓ Require status checks to pass

✓ Require branches to be up to date before merging
✓ Dismiss stale pull request approvals: SIM
```

### Branch: `hotfix/*` (Emergências)

```
Branch name pattern: hotfix/*

✓ Require a pull request before merging
  ✓ Require approvals: 2 reviewers
  ✓ Require the most recent push to be approved

✓ Restrict who can push: Maintainers
```

---

## 🔐 Segurança

### Settings → Code Security & Analysis

```
✓ Dependabot alerts: ENABLE
✓ Dependabot security updates: ENABLE
✓ Dependabot version updates: ENABLE
✓ Secret scanning: ENABLE (se disponível)
✓ Push protection: ENABLE (se disponível)
```

### Settings → Secrets and Variables → Actions

Adicione secrets (se usar CI/CD):

```
PYPI_API_TOKEN=...        # Para publicar em PyPI
DOCKER_HUB_TOKEN=...      # Para Docker
CODECOV_TOKEN=...         # Para cobertura
```

---

## 🤖 Automação (GitHub Actions)

Todos os workflows já foram criados em `.github/workflows/`:

- ✅ `tests.yml` - Testes em todo PR
- ✅ `stale.yml` - Fechar issues inativas
- ✅ `label.yml` - Auto-labeling
- ✅ `release.yml` - Release automático

Nada a configurar aqui! ✨

---

## 👥 Colaboradores

### Settings → Collaborators and teams

**Criar 3 Teams:**

#### Team 1: Maintainers

```
Name: libria/maintainers
Permission: Admin
Members: [Lead developers]
```

#### Team 2: Reviewers

```
Name: libria/reviewers
Permission: Maintain
Members: [Code reviewers]
```

#### Team 3: Contributors

```
Name: libria/contributors
Permission: Pull request access
Members: [Todos os contribuidores]
```

---

## 📊 GitHub Pages (Documentação)

### Habilitar

```
Settings → Pages

Source: Deploy from a branch
Branch: main (ou develop)
Folder: / (root) ou /docs
Theme: Minimal (ou preferência)
Enforce HTTPS: SIM
```

Depois de habilitado, sua documentação estará em:
```
https://Gabryel-lima.github.io/LibrIA/
```

---

## 📱 Integrações Externas

### CodeCov (Cobertura de Testes)

1. Acesse https://codecov.io
2. Conecte com GitHub
3. Ative para o repositório LibrIA
4. Adicione badge ao README:

```markdown
[![codecov](https://codecov.io/gh/Gabryel-lima/LibrIA/branch/develop/graph/badge.svg)](https://codecov.io/gh/Gabryel-lima/LibrIA)
```

### Snyk (Segurança)

1. Acesse https://snyk.io
2. Conecte GitHub
3. Autorize o repositório
4. Receberá PRs automáticas para vulnerabilidades

### SonarCloud (Qualidade)

1. Acesse https://sonarcloud.io
2. Conecte GitHub
3. Adicione ao workflow de CI/CD

---

## ✅ Community Profile

Verifique em: Settings → Community

Certifique-se que todos estão marcados ✓:

- ✅ Code of Conduct: CODE_OF_CONDUCT.md
- ✅ Contributing: CONTRIBUTING.md
- ✅ License: LICENSE
- ✅ README: README.md
- ✅ Issue templates: .github/ISSUE_TEMPLATE/
- ✅ Pull request template: .github/pull_request_template.md

Score alvo: 100% ⭐

---

## 📋 Estrutura de Branches

```
main (production)
  ↑
  └─ PRs do develop (com 2 reviews)

develop (staging/integration)
  ↑
  ├─ feat/new-feature
  ├─ fix/bug-name
  ├─ docs/improvements
  ├─ test/coverage
  └─ refactor/cleanup

hotfix (emergência)
  ├─ hotfix/critical-bug
  └─ Cherry-pick para main + develop
```

---

## 🚀 Workflow Recomendado

### Para Contribuidores

```bash
# 1. Fork e clone
git clone https://github.com/Gabryel-lima/LibrIA.git
cd LibrIA

# 2. Configure upstream
git remote add upstream https://github.com/Gabryel-lima/LibrIA.git

# 3. Crie feature branch
git checkout -b feat/amazing-feature

# 4. Commit com convenção
git commit -m "feat(module): description of change"

# 5. Push para seu fork
git push origin feat/amazing-feature

# 6. Abra PR para develop
# (não para main!)
```

### Para Maintainers

```bash
# 1. Review
# 2. Merge para develop
# 3. Antes de release:
git checkout develop
git pull upstream develop

# 4. Crie release
git checkout -b release/v1.2.0
# ... atualizar CHANGELOG, version files ...
git commit -m "chore: bump version to 1.2.0"

# 5. Merge para main
git checkout main
git merge --no-ff release/v1.2.0
git tag -a v1.2.0 -m "Release 1.2.0"

# 6. Merge tag volta para develop
git checkout develop
git merge --no-ff release/v1.2.0

# 7. Push
git push upstream main develop
git push upstream --tags
```

---

## 📋 Checklist Final

Antes de liberar o repositório:

- [ ] README.md completo e atualizado
- [ ] CONTRIBUTING.md com instruções claras
- [ ] CODE_OF_CONDUCT.md presente
- [ ] LICENSE definida (MIT)
- [ ] CHANGELOG.md criado
- [ ] ROADMAP.md definido
- [ ] GOVERNANCE.md explicando estrutura
- [ ] .github/ com templates e workflows
- [ ] Branch main protegido (2 reviews)
- [ ] Branch develop protegido (1 review)
- [ ] Labels criados
- [ ] Community profile 100%
- [ ] GitHub Actions workflows testados
- [ ] Dependabot configurado
- [ ] Discussions habilitado
- [ ] Issues templates funcionando
- [ ] PR template funcionando
- [ ] Descrição e topics no About
- [ ] Website linkado (se aplicável)
- [ ] Primeiro release tagueado (v1.0.0)

---

## 🎓 Recursos Úteis

- [GitHub Docs - Branch Protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semver - Semantic Versioning](https://semver.org/lang/pt_BR/)
- [Keep a Changelog](https://keepachangelog.com/)

---

## 🆘 Troubleshooting

### "Preciso fazer hotfix mas estou em feature branch"

```bash
# Stash suas mudanças
git stash

# Mude para main
git checkout main
git pull upstream main

# Crie hotfix
git checkout -b hotfix/critical-bug

# Faça fix e commit
# Quando pronto:
git push origin hotfix/critical-bug
# → Abra 2 PRs: uma para main, uma para develop
```

### "Quantos commits devo fazer?"

Use: `git rebase -i` para reorganizar commitspath antes de fazer merge.

Ideal: 1-3 commits bem estruturados por feature.

### "Violei a branch protection, e agora?"

Rebase e force-push:

```bash
git rebase origin/develop
git push origin feat/feature-name --force-with-lease
```

---

## 📞 Suporte

Se tiver dúvidas sobre configuração:

- 📧 Email: gabbryellimasi@gmail.com
- 💬 Discussions: [GitHub Discussions](https://github.com/Gabryel-lima/LibrIA/discussions)

---

**Pronto para receber contribuições! 🎉**

Atualizado: 2026-02-16
