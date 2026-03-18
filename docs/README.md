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

- Dataset documentado com artefatos espelhados `_mirror.npy` para coleta estática e temporal
- Fluxo híbrido documentado com comparação original + espelhado antes da arbitragem
- Guias de desenvolvimento e datasets atualizados com plots persistidos em `training_plots/`
- README principal e índices atualizados para refletir exit code confiável em `main.py`

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
make train-hybrid
make infer-hybrid
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

Última atualização: 2026-03-17
