# Documentação - LibrIA

Índice da pasta `docs/` com foco no fluxo atual do projeto.

## Leitura recomendada

| Objetivo | Documento |
|----------|-----------|
| Setup local | [DEVELOPMENT.md](DEVELOPMENT.md) |
| Datasets e artefatos | [DATASETS.md](DATASETS.md) |
| CPUs sem AVX | [AVX_COMPATIBILITY.md](AVX_COMPATIBILITY.md) |
| Atualizações recentes | [ATUALIZACOES.md](ATUALIZACOES.md) |
| Processo de PR | [PULL_REQUEST_GUIDE.md](PULL_REQUEST_GUIDE.md) |

## O que mudou nesta rodada

- Pipeline temporal com coleta em `dataset/sequences/`
- Treinamento e inferência LSTM via `main.py train_lstm` e `main.py infer_lstm`
- Fluxo de calibração de câmera com tabuleiro 9x6
- Extração de features configurável por `FEATURE_MODE`

## Estrutura da pasta

```text
docs/
├── README.md
├── ATUALIZACOES.md
├── AVX_COMPATIBILITY.md
├── DATASETS.md
├── DEVELOPMENT.md
├── PULL_REQUEST_GUIDE.md
├── video_format_changes.md
├── api/
│   ├── cam.md
│   └── linker_py.md
└── guides/
    └── README.md
```

## Comandos mais citados na documentação

```bash
make setup
make verify-setup
make collect
make process
make train
make infer
make collect-sequences
make train-lstm
make infer-lstm
make capture-calibration
```

## Referências na raiz

- [../README.md](../README.md)
- [../CHANGELOG.md](../CHANGELOG.md)
- [../CONTRIBUTING.md](../CONTRIBUTING.md)
- [../DOCUMENTATION_INDEX.md](../DOCUMENTATION_INDEX.md)

Última atualização: 2026-03-10
