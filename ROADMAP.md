# 🗺️ Roadmap - LibrIA

Visão geral do desenvolvimento do LibrIA. Atualizado trimestralmente.

Legenda: ✅ concluído (com código no repositório) · 🟡 parcial · ⬜ não iniciado

## ✅ Concluído (v1.0.0)

- [x] Coleta de dados de gestos de Libras
- [x] Processamento e normalização de landmarks (`wrist_relative` e `bounding_box`)
- [x] Modelo Random Forest (`src/model_training/libras_model_trainer.py`; a acurácia de 99% citada na versão anterior deste roadmap não foi reverificada)
- [x] Inferência em tempo real via webcam
- [x] Documentação básica
- [x] Testes unitários

## ✅ Concluído desde a v1.0.0

- [x] Camada 1 de robustez: features `wrist_relative` (padrão) e `bounding_box` (legado)
- [x] Camada 2 de robustez: calibração de câmera (`scripts/calibrate_camera.py`)
- [x] Espelhamento esquerda/direita em datasets estáticos e temporais, com backfill de `frame_XXX.png` legados
- [x] Pipeline temporal robusto sobre o modelo LSTM (`src/inference/temporal_pipeline.py`)
- [x] Código de treino das CNNs embarcadas e do exportador de bundle (`src/model_training/libras_embedded_*`). A exportação TFLite int8 foi corrigida (via SavedModel, `src/model_training/tflite_export.py`) e coberta por teste; **ainda não foi rodada com o dataset real**, então não há `.tflite` nem métricas do modelo real
- [x] Cobertura de dataset e relatório de lacunas (`src/dataset/coverage.py`, `make report`)
- [x] Ingestão de datasets externos em vídeo (`src/dataset/video_ingest.py`, `make ingest`)
- [x] Metadados de proveniência por amostra (`src/dataset/sample_metadata.py`)
- [x] Divisão de dados por pessoa, para evitar vazamento entre treino/validação/teste (`src/evaluation/dataset_splits.py`)

## 🚧 Em Desenvolvimento (v1.1.0 — prazo original Q2 2026, atrasado)

- [ ] **Coleta para fechar a FASE 1** 🟡
  - Código pronto; falta a coleta em si (≥3 pessoas, variação de mão, luz, distância e câmera, e cobertura por classe conferida com `make report`)
  - Detalhes em `docs/FASE1_RECONHECIMENTO.md`, seção 6
  - Prioridade: Alta

- [ ] **App Android para testar o modelo no celular** 🟡
  - Plano em `.github/plans/android_plan.md` (fases F0–F5).
  - F1 concluída: núcleo C++ em `native/libria_core`, com paridade bit a bit contra o pipeline Python (`make native-test`).
  - F2 em andamento: ponte JNI e API Kotlin verificadas na JVM (`make native-bridge-test`); app Android (Kotlin, CameraX, MediaPipe) escrito, mas ainda não compilado (o ambiente bloqueia o Android SDK).
  - Depende de gerar os `.tflite` localmente (`make embedded-train`), o que exige o dataset local e TensorFlow. A exportação int8, que abortava, foi corrigida e testada com dados sintéticos; falta rodar com o dataset real e registrar as métricas.
  - Prioridade: Alta

- [ ] **Runtime embarcado para Raspberry Pi / Pico (C++)** ⬜ (depois do Android)
  - O exportador de bundle existe e funciona com dados sintéticos; `src/interfaces/libria_embedded_runtime.cpp` ainda tem os TODOs de `Init`, `PredictStatic` e `PredictTemporal` (retornam predição inválida)
  - Prioridade: Média, após o Android

- [ ] **Segurança de dependências: itens abertos** 🟡
  - Resolvido: 106 → 1 advisory na árvore completa (ver `docs/SEGURANCA.md`)
  - Aberto: protobuf 4.25.9 (exige migrar TensorFlow e MediaPipe, que fixam `protobuf<5`)
  - Feito: `requirements-gpu.txt` alinhado (`tensorflow[and-cuda]==2.16.1`; o torch 2.13 já traz o próprio CUDA). Não testado em GPU
  - Decisão pendente: remover os modelos legados `best_temporal_model.h5` e `temporal_cnn_model.h5` (sem uso, contrato 16 × 42 incompatível com o pipeline atual)
  - Feito: PR #18 do Dependabot fechado sem merge (propunha uma combinação torch/torchvision insolúvel)
  - Feito: numpy de volta a 1.26.4 (o bump para 2.5.4 no #22 era incompatível com o TensorFlow 2.16.1); o Dependabot ignora majors do numpy
  - Prioridade: Média

- [ ] **Face Recognition** - Reconhecimento de expressões faciais ⬜
  - Sem código no repositório
  - Estimado: 4-6 semanas
  - Prioridade: Alta

- [ ] **Multi-hand Support** - Melhorar detecção com múltiplas mãos ⬜
  - Os classificadores usam `max_num_hands=1`; nenhum tratamento de múltiplas mãos
  - Estimado: 2-3 semanas
  - Prioridade: Média

- [ ] **Tests Expansion** - Aumentar cobertura para 85%+ 🟡
  - 176 testes passam (`pytest tests`; 2 deles exigem TensorFlow e são pulados sem ele); a cobertura ainda não é medida
  - Só a exportação TFLite int8 exercita o TensorFlow (`tests/test_tflite_export.py`). Treino e inferência com modelos reais, Pillow e torch continuam sem teste
  - Estimado: 2 semanas
  - Prioridade: Alta

## 📋 Planejado (v1.2.0 — Q3 2026)

- [ ] **Web Interface** - Interface web interativa ⬜
  - Tecnologia: FastAPI + React
  - Status: Planning

- [ ] **TensorFlow Integration** - Suporte opcional para TensorFlow ⬜
  - Alternativa ao Random Forest
  - Status: Proposal

- [ ] **Model Compression** - Otimizar para edge devices 🟡
  - TF Lite, ONNX support
  - Status: Planning. A quantização int8 funciona (exportação corrigida, testada com dados sintéticos); não há suporte a ONNX.

- [ ] **Comparação de modelos com protocolo único** ⬜
  - Random Forest, LSTM, CNN e um futuro Transformer avaliados com o mesmo split por pessoa e as mesmas métricas por classe
  - Origem: `.github/plans/libras_robustez_3_camadas.md`, prioridade 4

## 🎯 Futura (v2.0.0)

- [ ] **Mobile App** - Aplicativo mobile nativo ⬜
  - iOS + Android
  - O app de teste Android (v1.1, acima) é um precursor, não o produto final

- [ ] **Real-time Translation** - Tradução em tempo real Libras ↔ Português ⬜

- [ ] **Community Dataset** - Crowdsourcing de dados de novos gestos ⬜

- [ ] **Plugin System** - Sistema de plugins para extensões ⬜

## 📊 Milestones

| Milestone | Data | Versão | Situação |
|-----------|------|--------|----------|
| Alpha Release | 2025-06 | v0.1.0 | ✅ |
| Beta Release | 2025-09 | v0.5.0 | ✅ |
| v1.0.0 | 2026-02 | v1.0.0 | ✅ |
| v1.1.0 | 2026-05 | v1.1.0 | ⚠️ atrasado |
| v1.2.0 | 2026-08 | v1.2.0 | ⚠️ atrasado |

As datas de Alpha e Beta foram mantidas como registro histórico. Não verifiquei tags de release para confirmá-las.

## 🎓 Community Goals

- Atingir 100 contribuidores
- Treinar 50 pessoas em ML & Computer Vision
- Integração com 5+ organizações de acessibilidade

## 💬 Feedback da Comunidade

Sua opinião importa! Como você gostaria que o LibrIA evoluísse?

- 👍 Vote em issues/features no GitHub
- 💬 Participe das discussões
- 📧 Contacte gabbryellimasi@gmail.com

---

**Última atualização**: 2026-11-01

Para mais detalhes, veja [GitHub Projects](https://github.com/Gabryel-lima/LibrIA/projects)
