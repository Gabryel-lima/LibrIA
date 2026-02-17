# GitHub Settings Recommendations

Recomendações de configuração para o repositório no GitHub.

## 🏠 General Settings

### Repository Details

- **Description**: "Sistema completo de reconhecimento de gestos de Libras com visão computacional"
- **Website**: `https://seu-site.com` (opcional)
- **Topics**: `libras`, `sign-language`, `computer-vision`, `machine-learning`, `python`, `asl`, `gesture-recognition`

### Default Branch

- Definir como **`develop`** (não `main`)
- `main` é apenas para releases

## 🔒 Security & Access

### Repository Visibility

- ✅ **Public** - Code aberto, mas requer review antes de merge

### Branch Protection Rules

#### Rule 1: `main` (Production)

```
Branch name pattern: main

Require pull request reviews before merging:
  ☑ Dismiss stale pull request approvals: SIM
  ☑ Require review from Code Owners: SIM
  ☑ Number of approvals required: 2

Require status checks to pass before merging:
  ☑ Require branches to be up to date before merging
  Status checks that must pass:
    - ✓ Build and Test (CI/CD)
    - ✓ Code Quality (Pylint)
    - ✓ Test Coverage (80%+)

Require code reviews before merging:
  ☑ Allow auto-merge: NÃO
  ☑ Dismiss stale pull request approvals: SIM
```

#### Rule 2: `develop` (Development)

```
Branch name pattern: develop

Require pull request reviews before merging:
  ☑ Number of approvals required: 1
  ☒ Dismiss stale pull request approvals: SIM

Require status checks to pass before merging:
  ✓ Build and Test
  ✓ Code Quality

Allow auto-merge: NÃO
```

#### Rule 3: `hotfix/*` (Emergency Fixes)

```
Branch name pattern: hotfix/*

Require pull request reviews before merging:
  ☑ Number of approvals required: 2 (mais rígido!)

Require status checks to pass

Restrict who can push to matching branches:
  ☑ Restrict pushes to: Mantenedores apenas
```

### Code Security & Analysis

#### GitHub Advanced Security (se disponível)

- ✅ **Dependabot alerts** - Avisos de vulnerabilidades
- ✅ **Dependabot security updates** - PRs automáticos
- ✅ **Secret scanning** - Detectar secrets vazados
- ✅ **Code scanning** - Análise estática

#### Archive warnings

- ✅ Ativar avisos de vulnerabilidades observadas

## 👥 Collaborators & Teams

### Teams a Criar

**Team 1: Maintainers**
- Permissão: Admin
- Pessoas: [Lead developers]

**Team 2: Reviewers**
- Permissão: Maintain
- Pessoas: [Code reviewers]

**Team 3: Contributors**
- Permissão: Pull request access
- Pessoas: [Contribuidores ativos]

### Invite Policy

- Require contributors assinem CLA (Contributor License Agreement) opcional
- Use bot GitHub para verificar CLA

## 📋 Issue Labels

Crie os seguintes labels para organizar issues:

| Label | Cor | Descrição |
|-------|-----|-----------|
| `bug` | 🔴 `d73a4a` | Algo não está funcionando |
| `enhancement` | 🟢 `a2eeef` | Nova funcionalidade |
| `documentation` | 📚 `0075ca` | Melhorias em docs |
| `good-first-issue` | 🟡 `7057ff` | Bom para iniciantes |
| `help-wanted` | 🟠 `ff6d00` | Precisa de ajuda |
| `question` | 🔵 `00a4d6` | Dúvida ou pergunta |
| `wontfix` | ⚫ `000000` | Não será resolvido |
| `duplicate` | 🟣 `cfd3d7` | Issue duplicada |
| `rfc` | 🟡 `7057ff` | Request for Comments |
| `high-priority` | 🔴 `ff0000` | Prioridade alta |
| `low-priority` | 🟢 `00ff00` | Prioridade baixa |
| `blocked` | 🔴 `ff0000` | Bloqueada por outra issue |
| `in-progress` | 🟠 `ffaa00` | Sendo trabalhada |
| `performance` | 🚀 `d4af37` | Performance improvement |
| `security` | 🔒 `dc143c` | Issue de segurança |

## 🔄 Merge Strategy

**Recomendado**: Squash and merge

Configurar em: Settings → General → Pull Request

```
Allow merge commits: ✗ (desabilitar)
Allow squash merging: ✓ (habilitar)
  Default commit message: "Pull request title and description"
Allow rebase merging: ✗ (desabilitar)
  
Automatically delete head branches: ✓ (sim)
```

## 📊 GitHub Pages (Documentação do Site)

### Habilitação

```
Settings → Pages

Source: Deploy from a branch
Branch: main
Folder: / (root)

Custom domain: docs.libria.dev (opcional)
Enforce HTTPS: ✓
```

### Template Sugerido

Usar Jekyll theme: "Minimal"

## 🤖 Automação com GitHub Actions

### Workflows Recomendados

1. **CI/CD Pipeline** - Testes em todo PR
2. **Code Quality** - Pylint, Black, mypy
3. **Security Scan** - Verificar vulnerabilidades
4. **Auto-labeling** - Classificar PRs automaticamente
5. **Stale Issues** - Fechar issues inativas
6. **Release Drafter** - Rascunar release notes automaticamente

## 📱 Integrações Externas

### Recomendadas

| Ferramenta | Propósito | Config |
|-----------|----------|--------|
| **CodeCov** | Cobertura de testes | Badge em README |
| **Snyk** | Seg. de dependências | Alertas |
| **SonarCloud** | Análise de código | Reports |
| **Dependabot** | Atualizar deps | PRs automáticas |
| **All Contributors Bot** | Reconhecer maintainers | Automático |

## 📝 Wikis & Documentation

- Desabilitar Wiki padrão do GitHub
- Usar documentação em `docs/` gitignored
- Hospedar em GitHub Pages

## 🎯 Milestones

Criar para releases planejadas:

```
Milestone: v1.1.0
Description: Adicionar face recognition
Due Date: [data]
```

## 🔔 Notification Settings

Recomendado para Maintainers:

- Email para reviews: SIM
- Email para mentions: SIM
- Email para CI failures: SIM

## 🚀 Deployment

### Environment Protection Rules

Se usar GitHub Deployments:

```
Main (production)
  Required reviewers: Todos os maintainers
  Timeout: 30 dias

Staging (develop)
  Required reviewers: 1 reviewer
  Timeout: 7 dias
```

## 🔐 Secrets & Variables

Armazenar em Settings → Secrets and variables → Actions:

```
PYPI_API_TOKEN - Para publicar em PyPI
DOCKER_TOKEN - Para publicar imagens
COVERAGE_TOKEN - Para codecov
```

## 📋 Community Profile Checklist

Garantir que seu repositório tenha:

- ✅ README.md descritivo
- ✅ LICENSE (MIT recomendado)
- ✅ CONTRIBUTING.md
- ✅ CODE_OF_CONDUCT.md
- ✅ .github/pull_request_template.md
- ✅ .github/ISSUE_TEMPLATE/
- ✅ CHANGELOG.md

Verificar em: Settings → Code security and analysis → Community profile

---

**Aplique estas configurações gradualmente após criar o repo!**
