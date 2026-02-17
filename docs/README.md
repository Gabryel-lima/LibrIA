# 📚 Documentação - LibrIA

Guia completo de documentação e referências do projeto.

## 🚀 Comece Aqui

| Objetivo | Documento |
|----------|-----------|
| **Quer contribuir?** | [PULL_REQUEST_GUIDE.md](PULL_REQUEST_GUIDE.md) |
| **Setup local?** | [DEVELOPMENT.md](DEVELOPMENT.md) |
| **Como fazer git/commits?** | [PULL_REQUEST_GUIDE.md](PULL_REQUEST_GUIDE.md#🔄-processo-passo-a-passo) |
| **Não sabe por onde começar?** | [DEVELOPMENT.md](DEVELOPMENT.md) |

## 📖 Documentação Principal

### Para Desenvolvedores
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Setup local, testes, debugging
- **[PULL_REQUEST_GUIDE.md](PULL_REQUEST_GUIDE.md)** - Como fazer PRs de qualidade

### Para Usuários
- **[DATASETS.md](DATASETS.md)** - Datasets disponíveis e downloads
- **[AVX_COMPATIBILITY.md](AVX_COMPATIBILITY.md)** - Problemas com CPU sem AVX
- **[video_format_changes.md](video_format_changes.md)** - Processamento de vídeo

### Estrutura de Pastas
```
docs/
├── README.md (você está aqui)
├── DEVELOPMENT.md           # Setup para devs
├── PULL_REQUEST_GUIDE.md   # Como fazer PRs
├── DATASETS.md              # Dados
├── AVX_COMPATIBILITY.md     # CPU
├── video_format_changes.md  # Vídeo
└── guides/                  # Guias adicionais
    └── README.md
```

## 🔗 Links Importantes

- **[README Principal](../README.md)** - Overview do projeto
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - Diretrizes gerais
- **[CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md)** - Código de conduta
- **[GOVERNANCE.md](../GOVERNANCE.md)** - Estrutura de decisões
- **[ROADMAP.md](../ROADMAP.md)** - Futuro do projeto

## 💡 Dicas Rápidas

### Instalação Rápida
```bash
git clone https://github.com/Gabryel-lima/LibrIA.git
cd LibrIA
make setup
```

### Executar Pipeline
```bash
make collect    # Coletar dados
make process    # Processar
make train      # Treinar
make infer      # Inferência
```

### Contribuir
```
1. Leia: PULL_REQUEST_GUIDE.md
2. Fork: https://github.com/Gabryel-lima/LibrIA
3. Clone: seu fork
4. Branch: git checkout -b feat/seu-feature
5. Commit: Siga Conventional Commits
6. PR: Para develop branch
```

## 🗂️ Organização de Documentos

### Na Raiz (/)
- `README.md` - Overview principal
- `CONTRIBUTING.md` - Como contribuir
- `CODE_OF_CONDUCT.md` - Código de conduta
- `GOVERNANCE.md` - Estrutura
- `ROADMAP.md` - Futuro
- `CHANGELOG.md` - Histórico
- `LICENSE` - MIT

### Em docs/
- `DEVELOPMENT.md` - Setup local
- `PULL_REQUEST_GUIDE.md` - Guia de PRs
- `DATASETS.md` - Dados
- `AVX_COMPATIBILITY.md` - CPU
- `video_format_changes.md` - Vídeo

### Em .github/
- `pull_request_template.md` - Template de PR
- `ISSUE_TEMPLATE/` - Templates de issues
- `workflows/` - GitHub Actions
- `archive/` - Documentação antiga (referência)

## 📞 Dúvidas?

- 📖 Leia a documentação relevante acima
- 💬 Abra uma [issue](https://github.com/Gabryel-lima/LibrIA/issues) com label `question`

---

Última atualização: 2026-02-17
