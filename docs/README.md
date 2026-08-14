# Documentação — LibrIA

Índice da pasta `docs/`.

## Leitura recomendada

| Objetivo | Documento |
|:---------|:----------|
| Vocabulário, metadados e avaliação por pessoa | [FASE1_RECONHECIMENTO.md](FASE1_RECONHECIMENTO.md) |
| Pipeline temporal, segmentação e `SignToken` | [FASE2_TEMPORAL.md](FASE2_TEMPORAL.md) |
| Arquitetura por diagramas | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Formatos de dataset e artefatos | [DATASETS.md](DATASETS.md) |
| Setup de desenvolvimento | [DEVELOPMENT.md](DEVELOPMENT.md) |
| CPUs sem AVX | [AVX_COMPATIBILITY.md](AVX_COMPATIBILITY.md) |
| Processo de Pull Request | [PULL_REQUEST_GUIDE.md](PULL_REQUEST_GUIDE.md) |
| Fórmulas da arquitetura (PDF) | [latex/model_architecture_equations.pdf](latex/model_architecture_equations.pdf) |

O plano arquitetural que orienta as fases fica em
[`.github/plans/atualização_projeto.md`](../.github/plans/atualização_projeto.md).

## Estrutura da pasta

```text
docs/
├── README.md                   este índice
├── FASE1_RECONHECIMENTO.md
├── FASE2_TEMPORAL.md
├── ARCHITECTURE.md
├── DATASETS.md
├── DEVELOPMENT.md
├── AVX_COMPATIBILITY.md
├── PULL_REQUEST_GUIDE.md
├── guides/
└── latex/
```

## Comandos citados na documentação

Todo alvo do Makefile tem um comando de mesmo nome no `main.py`
(`make train-temporal` == `python main.py train-temporal`).

```bash
make setup            # ambiente
make verify           # validação do ambiente
make collect          # dataset mínimo (alfabeto estático + J/Z)
make collect-words    # palavras e gestos funcionais
make collect-unknown  # classe de rejeição
make report           # cobertura do dataset
make train            # modelos estático + temporal
make infer            # inferência híbrida
make embedded-train   # CNNs quantizadas + bundle
make embedded-check   # validação do bundle
make test             # suíte de testes
```

Lista completa: `make help`.

## Referências na raiz

- [../README.md](../README.md)
- [../CHANGELOG.md](../CHANGELOG.md)
- [../CONTRIBUTING.md](../CONTRIBUTING.md)
- [../ROADMAP.md](../ROADMAP.md)
