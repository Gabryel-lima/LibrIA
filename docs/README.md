# Documentação - LibrIA

Índice da pasta `docs/` com foco no fluxo atual do projeto.

## Leitura recomendada

| Objetivo | Documento |
|----------|-----------|
| Entender a arquitetura por diagramas | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Setup local | [DEVELOPMENT.md](DEVELOPMENT.md) |
| Datasets e artefatos | [DATASETS.md](DATASETS.md) |
| CPUs sem AVX | [AVX_COMPATIBILITY.md](AVX_COMPATIBILITY.md) |
| Atualizações recentes | [ATUALIZACOES.md](ATUALIZACOES.md) |
| Processo de PR | [PULL_REQUEST_GUIDE.md](PULL_REQUEST_GUIDE.md) |

## O que mudou nesta rodada

- Guia visual novo com diagramas menores e navegação por etapa em `ARCHITECTURE.md`
- Documentação sincronizada com o pipeline embedded atual e export para Pico
- Guia de desenvolvimento reescrito para refletir os comandos e artefatos reais
- Guia de datasets ampliado com artefatos embedded e bundle exportável

## Estrutura da pasta

```text
docs/
├── ARCHITECTURE.md
├── README.md
├── ATUALIZACOES.md
├── AVX_COMPATIBILITY.md
├── DATASETS.md
├── DEVELOPMENT.md
├── PULL_REQUEST_GUIDE.md
├── video_format_changes.md
└── guides/
    └── README.md
```

## Comandos mais citados na documentação

```bash
make setup
make verify-setup
make collect-static
make collect-temporal
make collect-minimal-dataset
make train
make infer
make train-lstm
make infer-lstm
make train-embedded-all
make export-embedded
make infer-embedded
make capture-calibration
```

## Referências na raiz

- [../README.md](../README.md)
- [../CHANGELOG.md](../CHANGELOG.md)
- [../CONTRIBUTING.md](../CONTRIBUTING.md)
- [../DOCUMENTATION_INDEX.md](../DOCUMENTATION_INDEX.md)

Última atualização: 2026-03-12
