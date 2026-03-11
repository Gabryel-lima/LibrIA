# Índice de Documentação - LibrIA

Referência rápida para localizar a documentação alinhada ao estado atual do projeto.

## Por caso de uso

### Quero rodar o projeto
- [README.md](README.md)
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)

### Quero configurar ambiente local
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
- [docs/AVX_COMPATIBILITY.md](docs/AVX_COMPATIBILITY.md)

### Quero entender os dados e modelos
- [docs/DATASETS.md](docs/DATASETS.md)
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
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): setup, testes e fluxo de desenvolvimento
- [docs/DATASETS.md](docs/DATASETS.md): datasets, artefatos e formatos
- [docs/AVX_COMPATIBILITY.md](docs/AVX_COMPATIBILITY.md): limitações de CPU e bibliotecas com AVX
- [docs/ATUALIZACOES.md](docs/ATUALIZACOES.md): resumo das atualizações recentes
- [docs/PULL_REQUEST_GUIDE.md](docs/PULL_REQUEST_GUIDE.md): processo de PR

### Documentação técnica complementar
- [docs/api/cam.md](docs/api/cam.md): notas sobre captura em C++ e OpenCV
- [docs/api/linker_py.md](docs/api/linker_py.md): referência adicional
- [src/models/transformer-gpt/README.md](src/models/transformer-gpt/README.md): modelo experimental para sequências

## Fluxo recomendado

1. Leia [README.md](README.md) para entender o pipeline estático e temporal.
2. Faça o setup com [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).
3. Se a máquina tiver CPU antiga, confira [docs/AVX_COMPATIBILITY.md](docs/AVX_COMPATIBILITY.md) antes de instalar tudo.
4. Para dados e artefatos, use [docs/DATASETS.md](docs/DATASETS.md).
5. Para contribuir, siga [docs/PULL_REQUEST_GUIDE.md](docs/PULL_REQUEST_GUIDE.md).

## Checklist rápido

- [ ] Li [README.md](README.md)
- [ ] Rodei `make setup`
- [ ] Rodei `make verify-setup`
- [ ] Sei qual fluxo vou usar: estático ou temporal
- [ ] Verifiquei restrições de AVX quando necessário

## Última atualização

- Data: 2026-03-10
- Status: documentação sincronizada com o dataset unificado, calibração de câmera e feature extraction configurável
