# Índice de Documentação - LibrIA

Referência rápida para localizar a documentação alinhada ao estado atual do projeto.

## Por caso de uso

### Quero rodar o projeto
- [README.md](README.md)
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

### Quero configurar ambiente local
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
- [docs/AVX_COMPATIBILITY.md](docs/AVX_COMPATIBILITY.md)

### Quero entender os dados e modelos
- [docs/DATASETS.md](docs/DATASETS.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [README.md](README.md)

### Quero contribuir
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/PULL_REQUEST_GUIDE.md](docs/PULL_REQUEST_GUIDE.md)

### Quero ver o histórico recente
- [CHANGELOG.md](CHANGELOG.md)
- [docs/ATUALIZACOES.md](docs/ATUALIZACOES.md)

### Quero referência técnica adicional
- [docs/video_format_changes.md](docs/video_format_changes.md)
- [src/models/transformer-gpt/README.md](src/models/transformer-gpt/README.md)

## Estrutura da documentação

### Raiz
- [README.md](README.md): visão geral, fluxos principais e comandos
- [CHANGELOG.md](CHANGELOG.md): histórico de mudanças
- [CONTRIBUTING.md](CONTRIBUTING.md): diretrizes de contribuição
- [ROADMAP.md](ROADMAP.md): direção futura do projeto

### Pasta docs
- [docs/README.md](docs/README.md): índice da pasta docs
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): arquitetura visual em diagramas curtos e navegáveis
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): setup, testes e fluxo de desenvolvimento
- [docs/DATASETS.md](docs/DATASETS.md): datasets, artefatos e formatos
- [docs/AVX_COMPATIBILITY.md](docs/AVX_COMPATIBILITY.md): limitações de CPU e bibliotecas com AVX
- [docs/ATUALIZACOES.md](docs/ATUALIZACOES.md): resumo das atualizações recentes
- [docs/PULL_REQUEST_GUIDE.md](docs/PULL_REQUEST_GUIDE.md): processo de PR

### Documentação técnica complementar
- [src/models/transformer-gpt/README.md](src/models/transformer-gpt/README.md): modelo experimental para sequências

## Fluxo recomendado

1. Leia [README.md](README.md) para visão geral e comandos principais.
2. Use [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) para navegar pelos fluxos em diagramas curtos.
3. Faça o setup com [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).
4. Se a máquina tiver CPU antiga, confira [docs/AVX_COMPATIBILITY.md](docs/AVX_COMPATIBILITY.md) antes de instalar tudo.
5. Para dados e artefatos, use [docs/DATASETS.md](docs/DATASETS.md).
6. Para contribuir, siga [docs/PULL_REQUEST_GUIDE.md](docs/PULL_REQUEST_GUIDE.md).

## Checklist rápido

- [ ] Li [README.md](README.md)
- [ ] Li [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [ ] Rodei `make setup`
- [ ] Rodei `make verify-setup`
- [ ] Sei qual fluxo vou usar: estático ou temporal
- [ ] Verifiquei restrições de AVX quando necessário

## Última atualização

- Data: 2026-03-12
- Status: documentação sincronizada com o pipeline embedded atual, com diagramas de navegação por etapa
