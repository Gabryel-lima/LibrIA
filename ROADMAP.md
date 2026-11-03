# 🗺️ Roadmap - LibrIA

Visão geral do desenvolvimento do LibrIA. Atualizado trimestralmente.

Este é o to-do do projeto: lista só o que **falta**. O que foi concluído está no [HISTORICO.md](HISTORICO.md).

Legenda: 🟡 parcial · ⬜ não iniciado

## ✅ Concluído

Itens concluídos saem deste arquivo e ficam registrados no [HISTORICO.md](HISTORICO.md),
com o motivo, a verificação e o commit. Itens parciais permanecem aqui com o que
falta e um link `→ H-NNN` para a parte já feita. A regra completa está no topo do
histórico.

- Antes de 2026-11-03 (v1.0.0 e o que veio depois): [H-000](HISTORICO.md#h-000--itens-concluídos-antes-deste-histórico-migrados-do-roadmap)

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
  - Já feito: 106 → 1 advisory, `requirements-gpu.txt`, PR #18 e numpy → [H-000](HISTORICO.md#h-000--itens-concluídos-antes-deste-histórico-migrados-do-roadmap)
  - Aberto: protobuf 4.25.9. Com `tensorflow==2.19.1` (aceita `protobuf<6`), o único bloqueio é o `mediapipe==0.10.14` (`protobuf<5`, API legada `mp.solutions`) → [H-003](HISTORICO.md#h-003--correções-de-documentação-sobre-tensorflow-2191-e-protobuf)
  - Decisão pendente: remover os modelos legados `best_temporal_model.h5` e `temporal_cnn_model.h5` (sem uso, contrato 16 × 42 incompatível com o pipeline atual)
  - Não testado: `tensorflow[and-cuda]==2.19.1` em GPU
  - Prioridade: Média

- [ ] **Face Recognition** - Reconhecimento de expressões faciais ⬜
  - Sem código no repositório
  - Estimado: 4-6 semanas
  - Prioridade: Alta

- [ ] **Multi-hand Support** - Melhorar detecção com múltiplas mãos 🟡
  - Já feito: com duas mãos no quadro, o sistema escolhe e segue a mão certa (`utils/hand_selection.py`) → [H-001](HISTORICO.md#h-001--seleção-da-mão-certa-quando-há-mais-de-uma-mão-no-quadro)
  - Falta: testar com webcam e MediaPipe real, e medir o custo de `max_num_hands=2`
  - Falta: sinais feitos com as duas mãos (exige mudar o contrato de features para 2 × 63 e retreinar)
  - Prioridade: Média

- [ ] **Tests Expansion** - Aumentar cobertura para 85%+ 🟡
  - Cobertura medida: **44,4%** com `make test-coverage` (200 testes; 2 exigem TensorFlow e são pulados sem ele) → [H-002](HISTORICO.md#h-002--medição-de-cobertura-de-testes)
  - Maiores buracos: Transformer experimental (0%), scripts de câmera e fetch (0%), `libras_realtime_classifier.py` (11%), `main.py` (28%)
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

**Última atualização**: 2026-11-03

Para mais detalhes, veja [GitHub Projects](https://github.com/Gabryel-lima/LibrIA/projects)
