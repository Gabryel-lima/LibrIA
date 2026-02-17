# 🏛️ Governança do Projeto LibrIA

## Estrutura de Liderança

### Core Team (Mantenedores)

| Papel | Responsabilidade |
|------|------------------|
| **Maintainer Principal** | Visão geral, decisões arquiteturais, releases |
| **Maintainers** | Review de PRs, triagem de issues |
| **Contributors Ativos** | Desenvolvimento, testes, documentação |

## Processo de Decisão

### Decisões Menores

- **PRs de bug fixes e documentação**: Aprovação de 1 maintainer
- **Melhorias pequenas**: 1 review + testes passando
- **Timeframe**: Máximo 7 dias

### Decisões Maiores

- **Mudanças arquiteturais**: Discussão em Discussions/RFC
- **Novas features**: 2 reviews + comunidade input
- **Breaking changes**: RFC + votação
- **Timeframe**: Mínimo 14 dias de discussão

### RFC (Request for Comments)

Para mudanças significativas:

1. Crie issue com label `rfc`
2. Descreva proposta, motivação, alternativas
3. Aguarde feedback (mínimo 2 semanas)
4. Proceeda com implementação

**Template RFC:**

```markdown
# RFC: [Título Descritivo]

## Motivação
Por que essa mudança é necessária?

## Proposta
Descrição detalhada da mudança.

## Alternativas Consideradas
O que mais foi considerado?

## Riscos
Quais são os riscos potenciais?

## Implementação
Como isso será implementado?
```

## Strategy Linear (Branch Strategy)

### Branches Principais

| Branch | Propósito | Política |
|--------|-----------|----------|
| **main** | Produção, releases estáveis | Protected, requer review + tests |
| **develop** | Desenvolvimento principal | Protected, staging |
| **feature*** | Novas funcionalidades | Temporários, deletados após merge |
| **hotfix*** | Correções críticas | Cherry-pick para main e develop |

### Workflow

```
feature/new-feature
       ↓
   Pull Request (develop branch)
       ↓
   Code Review + Tests
       ↓
   Merge to develop
       ↓
  [Quando pronto]
       ↓
   Release Branch / PR para main
       ↓
   Deploy / Tag
```

## Política de Releases

### Versionamento

Seguimos [Semantic Versioning](https://semver.org/lang/pt_BR/):

```
MAJOR.MINOR.PATCH
```

- **MAJOR**: Mudanças incompatíveis 
- **MINOR**: Novas funcionalidades compatíveis
- **PATCH**: Correções de bugs

**Exemplo**: `1.2.3` = Major 1, Minor 2, Patch 3

### Ciclo de Release

1. **Desenvolvimento** - Features em `develop`
2. **Feature Freeze** - Congelamento por 1 semana
3. **Release Candidate** - Branch `release/v1.2.0`
4. **Beta Testing** - Testes intensivos
5. **Release** - Tag e merge para `main`

### Changelog

Manter [CHANGELOG.md](CHANGELOG.md) atualizado:

```markdown
## [1.2.0] - 2026-02-16

### Added
- Nova funcionalidade de face recognition
- Suporte para múltiplas hands

### Fixed
- Bug em frame capture
- Problema de memória em inferência

### Changed
- Melhorada documentação
- Otimizada pipeline de processamento
```

## Critérios de Merge

Um PR pode ser mergeado quando:

- ✅ Testes passam (coverage ≥80%)
- ✅ Sem conflitos com base branch
- ✅ Aprovado por pelo menos 1 maintainer
- ✅ Código segue padrões do projeto
- ✅ Documentação atualizada
- ✅ Commits bem organizados

## Reconhecimento

Colaboradores são reconhecidos:

- No [CONTRIBUTORS.md](CONTRIBUTORS.md)
- Em release notes
- No README.md (top contributors)

## Roadmap

O roadmap é publicado em:

- [GitHub Projects](https://github.com/projeto/LibrIA/projects)
- [ROADMAP.md](docs/ROADMAP.md)
- Atualizado a cada trimestre

## Comunicação

### Canais Oficiais

- 📧 **Email**: gabbryellimasi@gmail.com
- 💬 **GitHub Discussions**: [Link]
- 🎮 **Discord**: [Link]

### Reuniões Comunitárias

- **Bi-weekly**: Chamadas públicas com mantenedores
- **Monthly**: Review de roadmap
- **Ad-hoc**: RFC discussions

## Transição de Liderança

Se um maintainer precisar se afastar:

1. Notifica outros mantenedores com antecedência
2. Transfere pendências gradualmente
3. Documentar conhecimento crítico
4. Promover contributors experientes

## Conflito de Interesse

Mantenedores devem declarar:

- Profissão/empresa que pode ter interesse
- Projetos comerciais relacionados
- Relações pessoais relevantes

## Aprovação de Fundadores

Certas decisões requerem aprovação do founder/criador:

- Mudanças em modelo de governança
- Transferência de propriedade
- Questões legais

---

**Última atualização**: 2026-02-16

Para questões sobre governança, abra uma issue ou discussão.
