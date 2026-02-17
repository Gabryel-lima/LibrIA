# ⚙️ Recomendações Finais para GitHub

Checklist priorizado de configurações recomendadas para seu repositório LibrIA no GitHub.

---

## 🔴 CRÍTICO (Faça Primeiro!)

### 1. Branch Protection: `main`

```
Settings → Branches → Add rule

Branch name pattern: main

✓ Require a pull request before merging
  ✓ Dismissals of stale pull request approvals: true
  ✓ Require approval from code owners: true
  ✓ Required number of approvals before merging: 2

✓ Require status checks to pass before merging
  ✓ Require branches to be up to date before merging

✓ Require conversation resolution before merging

✓ Restrict who can push to matching branches
  Allowed to push: Only the following
  → Select: Maintainers team
```

### 2. Branch Protection: `develop`

```
Branch name pattern: develop

✓ Require a pull request before merging
  ✓ Dismissals of stale pull request approvals: true
  ✓ Required number of approvals before merging: 1

✓ Require status checks to pass before merging
```

### 3. PR Merge Strategy

```
Settings → General → Pull Requests

□ Allow merge commits: DESABILITAR
✓ Allow squash merging: HABILITAR
□ Allow rebase merging: DESABILITAR
✓ Automatically delete head branches: HABILITAR
```

**Por quê?**
- Squash mantém histórico linear
- Auto-delete evita branches órfãs
- Merge commits criam confusão

---

## 🟠 ALTO (Faça Hoje)

### 4. Segurança Configuração

```
Settings → Code security and analysis

✓ Dependabot alerts: ENABLE
✓ Dependabot security updates: ENABLE
✓ Secret scanning: ENABLE (se disponível)
✓ Private vulnerability reporting: ENABLE
```

### 5. Criar Labels

Via Labels ou use script:

```yaml
# Labels recomendados
type/bug: d73a4a (vermelho)                 [Bug fix]
type/feature: a2eeef (ciano)                [Nova feature]
type/docs: 0075ca (azul)                    [Documentação]
priority/high: ff0000 (vermelho escuro)     [Urgente]
priority/low: 90ee90 (verde claro)          [Baixa prioridade]
status/blocked: ff0000                      [Bloqueada]
status/in-progress: ffaa00                  [Em trabalho]
good-first-issue: 7057ff (roxo)             [Para iniciantes]
help-wanted: ff6d00 (laranja)               [Precisa ajuda]
```

### 6. Habilitar Discussions

```
Settings → General → Features

✓ Discussions: HABILITAR
```

**Benefício:** Comunidade pode discutir ideias antes de issues.

---

## 🟡 MÉDIO (Faça Semana Seguinte)

### 7.Inverte Colaboradores para Teams

```
Settings → Collaborators and teams

Create teams:
1. libria/maintainers (Admin)
2. libria/reviewers (Maintain)  
3. libria/contributors (Pull request access)
```

### 8. Configurar Page (Documentação)

```
Settings → Pages

Source: Deploy from a branch
Branch: develop (branch com docs/)
Folder: / (root)

Theme: Minimal (ou escolha sua)
✓ Enforce HTTPS: SIM
```

### 9. Criar Primeira Release

```bash
git tag -a v1.0.0 -m "LibrIA v1.0.0 - First stable release"
git push origin v1.0.0

# GitHub Actions vai criar release automaticamente
```

### 10. Integração com CodeCov (Opcional mas Recomendado)

```
1. Vá para https://codecov.io
2. Sign in com GitHub
3. Ative repositório LibrIA
4. Adicione ao README:

![codecov](https://codecov.io/gh/Gabryel-lima/LibrIA/branch/develop/graph/badge.svg)
```

---

## 🟢 BAIXO (Depois)

### 11. SonarCloud (Qualidade de Código)

```
Vá para: https://sonarcloud.io
Sign in com GitHub
Projetos → Import → LibrIA
Integração automática
```

### 12. Snyk (Segurança de Dependências)

```
Vá para: https://snyk.io
Sign in com GitHub
Autorize acesso
Receberá PRs automáticas para vulnerabilidades
```

### 13. GitHub Pages Theme

Se usar Pages customizar tema:

```
Settings → Pages → Theme → escolha tema
```

---

## 📋 Ordem Recomendada de Implementação

```
1. Repositório criado no GitHub
   ↓
2. Fazer push com todos os arquivos
   ↓
3. Criar branch develop
   ↓
4. CRÍTICO (15 min)
   - Branch protection main + develop
   - PR merge strategy
   ↓
5. ALTO (30 min)
   - Segurança
   - Labels
   - Discussions
   ↓
6. MÉDIO (1h)
   - Teams
   - Primeiro PR teste
   - Release v1.0.0
   ↓
7. Baixo (ao longo do tempo)
   - CodeCov, SonarCloud, Snyk
```

---

## 🔄 Fluxo de Trabalho Recomendado

```
Novo Contribuidor
    ↓
Fork + Clone
    ↓
Cria branch: feat/feature-name
    ↓
Compila, Testa, Formata
    ↓
Cria PR para develop
    ↓
Bot auto-labels
    ↓
CI roda testes
    ↓
Reviewer aprova (mínimo 1)
    ↓
Merge para develop (squash)
    ↓
[Quando pronto para release]
    ↓
Release PR main
    ↓
2 reviews aprovam
    ↓
Merge para main
    ↓
Tag v1.X.X
    ↓
GitHub Actions cria release
```

---

## 🎯 Configuração Ideal Resumida

| Configuração | Setting | Benefício |
|---|---|---|
| **Main branch** | Protegido, 2 reviews, status checks | Qualidade |
| **Develop branch** | Protegido, 1 review, status checks | Rapidez + Qualidade |
| **Merge strategy** | Squash | Histórico limpo |
| **Auto-delete** | Sim | Housekeeping |
| **PR templates** | Obrigatório respostas | Melhor info |
| **Issue templates** | Bug, Feature, Docs | Consistência |
| **Labels** | Padroniizados | Organização |
| **Dependabot** | Security + Version updates | Segurança |
| **GitHub Actions** | 4 workflows | Automação |
| **Discussions** | Habilitado | Comunidade |

---

## ❌ Evite Fazer

- ❌ Não proteja `main` com `allow force push`
- ❌ Não use `allow merge commits` (cria poluição)
- ❌ Não use `allow rebase` sozinho sem merge
- ❌ Não coloque _secrets_ no código (use Secrets)
- ❌ Não deixe branches órfãs (auto-delete enabled!)
- ❌ Não misture develop com main
- ❌ Não aceite PR sem testes (CI checks!)

---

## 🚀 Boas Práticas

### Para PRs
- ✅ Descrever claramente o que foi mudado
- ✅ Reference issues relacionadas (`Closes #123`)
- ✅ Aguarde CI passar antes de reviewers
- ✅ Responda feedback construtivo
- ✅ Mantenha PRs focado em um tópico

### Para Issues
- ✅ Use labels apropriadass
- ✅ Seja específico na descrição
- ✅ Prove passo a passo para reproduzir bugs
- ✅ Linque issues relacionadas
- ✅ Feche quando resolvida

### Para Reviews
- ✅ Seja respeitoso e construtivo
- ✅ Destaque o positivo também
- ✅ Sugira melhorias (não ordene)
- ✅ Aprove quando estiver bom
- ✅ Não ignore comentários de outros

---

## 📞 Contatos GitHub Support

Se tiver dúvidas:

- 📧 GitHub Support: https://support.github.com
- 📚 Docs: https://docs.github.com
- 💬 Community Forum: https://github.com/orgs/community
- 🐛 Issues: https://github.com/github/feedback

---

## ✅ Verificação Final

Antes de anunciar "open source":

```
[ ] Branch protection em main (2 reviews)
[ ] Branch protection em develop (1 review)
[ ] Merge strategy: squash only
[ ] Auto-delete branches: ativado
[ ] PR template: presente e bom
[ ] Issue templates: bug + feature + docs
[ ] Labels: 15+ padrões criados
[ ] Dependabot: ativado
[ ] GitHub Actions: workflows rodando
[ ] Discussions: habilitado
[ ] Community profile: 90%+
[ ] README: excelente
[ ] CONTRIBUTING: completo
[ ] Primeiro PR teste: sucesso!
```

---

**Bom luck! 🚀**

Seu repositório será uma referência em open source! 💙

---

**Atualizado:** 2026-02-16  
**Para:** GitHub Ações e Configurações  
**Nível:** Iniciante → Intermediário
