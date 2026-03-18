# 📋 Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
e esse projeto adere ao [Semantic Versioning](https://semver.org/lang/pt_BR/).

## [Não Lançado]

### Added
- Pipeline unificado com coleta em `dataset/static/` e `dataset/temporal/`
- Treinamento LSTM via `main.py train_lstm` e inferência temporal via `main.py infer_lstm`
- Scripts de calibração de câmera e geração/exibição de tabuleiro
- Metadados de `feature_mode` persistidos no dataset processado e nos modelos
- Guia de arquitetura em `docs/ARCHITECTURE.md` com diagramas curtos e navegação por etapa
- Testes para exit code do `main.py`, espelhamento no dataset e utilitários de inferência híbrida

### Changed
- Extração de landmarks agora é configurável por `FEATURE_MODE`
- O pipeline estático agora lê `sample_XXX.npy` direto de `dataset/static/`
- O pipeline temporal agora usa `dataset/temporal/` como caminho principal
- Makefile ganhou comandos `collect-static`, `collect-temporal` e `collect-minimal-dataset`
- Documentação central sincronizada com os fluxos atuais do projeto, incluindo bundle embedded e export para Pico
- Coleta estática e temporal agora salvam variantes espelhadas `_mirror.npy` para suportar simetria esquerda/direita
- Treinos LSTM e embedded temporal aceitam fallback para o diretório temporal legado quando o caminho atual está vazio
- Trainers LSTM e embedded agora persistem gráficos de histórico em `training_plots/`

### Removed
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
