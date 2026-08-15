# 📋 Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
e esse projeto adere ao [Semantic Versioning](https://semver.org/lang/pt_BR/).

## [Não Lançado]

### Added
- **Dados externos sem coleta manual**: catálogo de bases públicas de Libras em
  `config/data_sources.py` (`make sources`), download automatizável do MINDS-Libras
  via API do Zenodo (`make fetch SOURCE=...`) e ingestão de vídeos/imagens para o
  formato do dataset em `src/dataset/video_ingest.py` (`make ingest`). A ingestão
  reamostra sequências para o comprimento da LSTM, filtra pelo vocabulário, é
  idempotente (`.ingest_state.json`) e grava proveniência (`source_dataset`,
  `source_uri`, `license`) nos metadados de cada amostra
- **Coleta dirigida por lacunas**: `src/dataset/coverage.py` calcula o que falta por
  classe; `make collect` imprime o plano e pula as classes que já atingiram a meta
  (use `--all-labels` para forçar). `make report` passa a mostrar o que falta, a
  origem das amostras e as classes com origem única
- **Fase 1 — vocabulário e avaliação**: `config/vocabulary.py` como fonte única do
  vocabulário (famílias `alphabet`, `lexical`, `functional` e a classe de rejeição
  `DESCONHECIDO`); metadados por amostra em `src/dataset/sample_metadata.py`
  (pessoa, câmera, ambiente, mão dominante, duração, qualidade); divisão
  treino/validação/teste **por pessoa** e métricas por classe com taxa de rejeição
  e latência em `src/evaluation/`; relatório de cobertura via `make report`
- **Fase 2 — pipeline temporal robusto**: buffer com carimbo de tempo, detecção de
  movimento, segmentação de início/fim com histerese, suavização de probabilidades,
  supressão de duplicatas e saída padronizada `SignToken`
  (`partial`/`final`/`rejected`) em `src/inference/`
- Coleta com metadados de sessão: `make collect SUBJECT=... ENVIRONMENT=...`
- Alvos `collect-words`, `collect-unknown` e `report`
- `make test` roda a suíte de testes de verdade (174 testes)
- Pipeline unificado com coleta em `dataset/static/` e `dataset/temporal/`
- Treinamento LSTM via `main.py train_lstm` e inferência temporal via `main.py infer_lstm`
- Scripts de calibração de câmera e geração/exibição de tabuleiro
- Metadados de `feature_mode` persistidos no dataset processado e nos modelos
- Guia de arquitetura em `docs/ARCHITECTURE.md` com diagramas curtos e navegação por etapa
- Testes para exit code do `main.py`, espelhamento no dataset e utilitários de inferência híbrida

### Changed
- **Comandos renomeados para o mesmo nome no Makefile e no `main.py`**
  (`make train-temporal` == `python main.py train-temporal`):
  `train-lstm`→`train-temporal`, `infer-lstm`→`infer-temporal`,
  `train-hybrid`→`train`, `infer-hybrid`→`infer`, `train`→`train-static`,
  `infer`→`infer-static`, `collect-minimal-dataset`→`collect`,
  `train-embedded-all`→`embedded-train`, `export-embedded`→`embedded-export`,
  `infer-embedded`→`embedded-check`, `verify-setup`→`verify`,
  `generate-checkerboard`→`checkerboard`, `capture-calibration`→`calibrate-capture`
- Variáveis de coleta simplificadas: `SUBJECT`, `CAMERA_ID`, `ENVIRONMENT`,
  `DOMINANT_HAND`, `CAMERA`, `SAMPLES`, `SEQUENCES`
- `verify` deixou de ser pré-requisito de todo alvo — os comandos deixaram de
  pagar ~10s de import a cada execução
- Inferência híbrida passou a usar o pipeline temporal segmentado no lugar da
  janela fixa deslizante; o fallback estático só opina com a mão parada e fica em
  silêncio por um cooldown após cada sinal temporal
- Dados temporais migrados de `dataset/sequences/` para `dataset/temporal/`
- Extração de landmarks agora é configurável por `FEATURE_MODE`
- O pipeline estático agora lê `sample_XXX.npy` direto de `dataset/static/`
- O pipeline temporal agora usa `dataset/temporal/` como caminho principal
- Makefile ganhou comandos `collect-static`, `collect-temporal` e `collect-minimal-dataset`
- Documentação central sincronizada com os fluxos atuais do projeto, incluindo bundle embedded e export para Pico
- Coleta estática e temporal agora salvam variantes espelhadas `_mirror.npy` para suportar simetria esquerda/direita
- Treinos LSTM e embedded temporal aceitam fallback para o diretório temporal legado quando o caminho atual está vazio
- Trainers LSTM e embedded agora persistem gráficos de histórico em `training_plots/`

### Removed
- `test_setup.py` (movia e apagava arquivos da raiz como efeito colateral e travava
  todo alvo do Makefile), `test_inference.py`, `update_hearders.py`
- `backup_old_files/` e `src/backup_tests/`
- `DOCUMENTATION_INDEX.md`, `ORGANIZATION_SUMMARY.md`, `docs/ATUALIZACOES.md` e
  `docs/video_format_changes.md` (duplicavam README e CHANGELOG)
- Fallback para o diretório temporal legado `dataset/sequences/`
- Alvos redundantes `run-lstm`, `status` e `update`
- Fluxos legados `process`, `collect_jz`, `src/data_collection/`, `src/data_processing/` e `scripts/collect_sequences.py`
- Artefatos experimentais órfãos `src/interfaces/cam.cpp`, `main_test.cpp`, `setup.py` e documentação C++ demonstrativa sem integração com o pipeline atual

### Fixed
- Fluxo de coleta temporal agora mantém a janela OpenCV responsiva durante espera e gravação
- Inferência estática passou a respeitar dimensionalidade do modelo e calibração opcional
- `main.py` agora retorna status não nulo quando comandos falham, permitindo automação confiável
- Inferência híbrida agora compara hipóteses originais e espelhadas antes de escolher a predição final por trilha

---

## [1.0.0] - 2026-02-16

### Added
- Sistema completo de reconhecimento de Libras
- Coleta de dados via webcam
- Processamento de landmarks com MediaPipe
- Modelo Random Forest com 99% de acurácia
- Inferência em tempo real
- Documentação completa
- Testes unitários (80% coverage)
- Suporte CPU e GPU via CUDA 12.4
- Compatibilidade com CPUs sem AVX

### Fixed
- Problema de memória em inferência contínua
- Race condition em coleta de frames
- Crash ao desconectar webcam

---

## [0.5.0] - 2025-09-15

### Added
- Beta release para testes comunitários
- Nova interface UI
- Suporte para dataset ASL

### Changed
- Melhorado pipeline de processamento
- Refatorado código de coleta de dados

---

## [0.1.0] - 2025-06-01

### Added
- Versão alpha inicial
- Prototipo funcional
- Dataset básico

---

## Guia para Contribuidores

Quando fizer um PR que será incluído no release:

1. Descreva sua mudança em uma seção apropriada
2. Siga o padrão: Added, Changed, Fixed, Removed, Security
3. Use infinitivo em terceira pessoa: "Add", "Fix", "Remove"
4. Reference issues quando aplicável: `(#123)`

### Template

```markdown
### Added
- Nova feature (#123)
- Documentação de setup

### Fixed
- Bug em inference (#456)

### Changed
- Refatoração de utils

### Removed
- Código deprecated

### Security
- Patch de vulnerabilidade
```

---

## Versioning

- **MAJOR** (x.0.0): Mudanças incompatíveis (breaking changes)
- **MINOR** (0.x.0): Novas features compatíveis
- **PATCH** (0.0.x): Bug fixes

Exemplo: `1.2.3`
- Major: 1
- Minor: 2  
- Patch: 3
