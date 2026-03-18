# Espelhamento Esquerda/Direita no Pipeline Híbrido - Implementation Plan
## 1. 🎯 Objective
Adicionar suporte consistente a mão esquerda/direita no pipeline híbrido, espelhando amostras estáticas e temporais para treino e permitindo backfill legado por frames já coletados.

## 2. 🏗️ Tech Strategy
- Pattern: augmentação determinística por espelhamento no eixo X de landmarks.
- State: padrão habilitado no treino; coleta/backfill com controle por função para manter compatibilidade.
- Constraints: sem quebrar shape de features em FEATURE_MODE atual; mudanças mínimas e compatíveis com datasets existentes.

## 3. 📂 File Changes
| Action | File Path | Brief Purpose |
|:-------|:----------|:--------------|
| MOD | `scripts/collect_dataset.py` | Gerar versão espelhada no backfill e utilitários de espelhamento |
| MOD | `src/model_training/libras_model_trainer.py` | Duplicar amostras estáticas com espelho durante carga |
| MOD | `src/model_training/libras_lstm_trainer.py` | Duplicar sequências temporais com espelho durante carga |
| MOD | `src/inference/libras_hybrid_realtime_classifier.py` | TTA simples (original+espelhado) para estático e temporal |
| MOD | `tests/test_collect_dataset.py` | Cobertura de backfill com espelhamento |
| MOD | `tests/test_static_dataset_loader.py` | Cobertura de augmentação estática espelhada |
| NEW | `tests/test_lstm_dataset_loader.py` | Cobertura de augmentação temporal espelhada |
| MOD | `tests/test_hybrid_realtime_classifier.py` | Cobertura de utilitários de espelhamento na inferência |

## 4. 👣 Execution Sequence
1. RED: criar/ajustar testes para falhar com o comportamento atual.
2. GREEN: implementar espelhamento mínimo para static/temporal/backfill.
3. GREEN: aplicar TTA simples na inferência híbrida.
4. REFACTOR: consolidar helper de espelhamento e manter legibilidade.
5. VERIFY: executar suíte de testes alvo e revisar diffs.

## 5. ✅ Verification Standards
- [ ] `pytest tests/test_collect_dataset.py -q`
- [ ] `pytest tests/test_static_dataset_loader.py -q`
- [ ] `pytest tests/test_lstm_dataset_loader.py -q`
- [ ] `pytest tests/test_hybrid_realtime_classifier.py -q`
- [ ] `pytest tests -q`
