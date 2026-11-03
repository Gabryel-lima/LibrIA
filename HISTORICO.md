# 🧾 Histórico de alterações do to-do

Registro do que saiu do [ROADMAP.md](ROADMAP.md) (o to-do do projeto) e por quê.
O `ROADMAP.md` lista só o que **falta**; quando um item (ou parte dele) fica
pronto, ele sai de lá e ganha uma entrada aqui, com link nos dois sentidos.

O [CHANGELOG.md](CHANGELOG.md) continua sendo o resumo por versão para quem usa o
projeto. Este arquivo é o diário técnico para quem mantém: o motivo da mudança,
o que foi verificado, o que **não** foi e o que ainda falta.

## Como registrar uma alteração

Regra para qualquer pessoa ou agente que mexer no to-do (também está no
[CLAUDE.md](CLAUDE.md) e no [CONTRIBUTING.md](CONTRIBUTING.md#roadmap-e-histórico)):

1. Ao concluir um item do `ROADMAP.md`, **remova-o** de lá. Se só uma parte ficou
   pronta, mantenha o item com o que falta e acrescente `→ [H-NNN](HISTORICO.md#<âncora>)`,
   onde a âncora é o título da entrada no formato do GitHub (minúsculas, espaços
   viram `-`, pontuação e o travessão somem; ex.: `#h-002--medição-de-cobertura-de-testes`).
2. Crie uma entrada aqui, **no topo da lista**, com o próximo número `H-NNN`.
3. Toda entrada tem: data, origem (o item do roadmap), o que mudou, por quê,
   arquivos, como foi verificado, limites (o que não foi testado) e o que falta.
4. Ligue a entrada ao código: commit(s), branch ou PR.
5. Não apague entradas antigas. Se algo registrado se mostrar errado, corrija com
   uma nota datada na própria entrada ou numa entrada nova.

Modelo:

```markdown
### H-NNN — Título curto
- **Data**: AAAA-MM-DD · **Origem**: ROADMAP › <item> · **Commits**: <hash/PR>
- **O que mudou**: ...
- **Por quê**: ...
- **Arquivos**: ...
- **Verificação**: ...
- **Limites**: ...
- **Pendente** (continua no ROADMAP): ...
```

---

## Entradas

### H-003 — Correções de documentação sobre TensorFlow 2.19.1 e protobuf
- **Data**: 2026-11-03 · **Origem**: ROADMAP › Segurança de dependências · **Commits**: [7b8a8e5](https://github.com/Gabryel-lima/LibrIA/commit/7b8a8e5) (comentário do Makefile) e o commit deste histórico
- **O que mudou**: o item de segurança do roadmap, o `docs/SEGURANCA.md` e um
  comentário do `Makefile` ainda citavam `tensorflow==2.16.1`, mas o commit
  `01b08f4` já fixou `tensorflow==2.19.1` nos dois requirements.
- **Por quê**: a troca muda o diagnóstico do protobuf. Conferido nos metadados do
  PyPI em 2026-11-03: `tensorflow==2.19.1` aceita `protobuf>=3.20.3,<6` e
  `numpy>=1.26,<2.2`; `mediapipe==0.10.14` exige `protobuf>=4.25.3,<5`.
  **O único bloqueio para corrigir o advisory do protobuf agora é o MediaPipe**,
  não mais o TensorFlow.
- **Observação aberta** (não alterada): o comentário do `requirements.txt` diz que
  o TensorFlow 2.19.1 e o MediaPipe 0.10.14 "exigem numpy<2". Pelos metadados,
  o TensorFlow aceita até numpy 2.1 e o MediaPipe não declara limite. A restrição
  real provavelmente é binária (wheels do MediaPipe 0.10.14 compilados contra
  numpy 1.x), mas isso é inferência e não foi testado. Manter `numpy==1.26.4`
  continua sendo o lado seguro.
- **Arquivos**: `ROADMAP.md`, `docs/SEGURANCA.md`, `Makefile`.
- **Verificação**: consulta à API JSON do PyPI para as duas versões.
- **Limites**: não instalei o TensorFlow 2.19.1 neste ambiente (Python 3.13, sem
  wheel); a compatibilidade em execução não foi verificada por mim.

### H-002 — Medição de cobertura de testes
- **Data**: 2026-11-03 · **Origem**: ROADMAP › Tests Expansion ("a cobertura ainda não é medida") · **Commits**: [7b8a8e5](https://github.com/Gabryel-lima/LibrIA/commit/7b8a8e5)
- **O que mudou**: alvo `make test-coverage` (pytest + pytest-cov, relatório no
  terminal e em `htmlcov/`) e `.coveragerc` medindo `src`, `scripts`, `config`,
  `utils` e `main.py`. `make clean` remove `htmlcov/` e `.coverage`.
- **Por quê**: a meta do roadmap é 85%, mas sem medir não dá para saber a
  distância nem onde investir. O `.coveragerc` **não** exclui módulos sem teste
  (ex.: o Transformer experimental): tirá-los do denominador inflaria o número.
- **Resultado**: **44,4%** (5171 instruções, 2876 sem execução), 200 testes
  passando e 2 pulados, em Python 3.13 **sem** TensorFlow e MediaPipe. Os
  maiores buracos: `src/models/transformer-gpt/` (0%, experimental), scripts de
  câmera e fetch (0%), `libras_realtime_classifier.py` (11%), `main.py` (28%).
  Os módulos de pipeline temporal, avaliação e metadados ficam entre 90% e 100%.
- **Limites**: com TensorFlow instalado, os 2 testes pulados rodam e o número
  sobe um pouco. O número vale para este ambiente, não é uma linha de base fixa.
- **Pendente** (continua no ROADMAP): chegar a 85%.

### H-001 — Seleção da mão certa quando há mais de uma mão no quadro
- **Data**: 2026-11-03 · **Origem**: ROADMAP › Multi-hand Support · **Commits**: [7b8a8e5](https://github.com/Gabryel-lima/LibrIA/commit/7b8a8e5)
- **Problema**: todos os pontos de captura usavam `max_num_hands=1` e liam
  `results.multi_hand_landmarks[0]`. Com uma segunda mão no quadro, o MediaPipe
  pode devolver a mão errada, e a ordem da lista não é estável entre quadros:
  numa janela temporal de 30 quadros, uma troca no meio corrompe a sequência.
- **O que mudou**:
  - `utils/hand_selection.py`: `HandSelector` com quatro estratégias (`first` =
    comportamento antigo, `largest`, `most_confident`, `dominant`) e
    rastreamento: com uma mão já escolhida, segue a candidata cujo punho está a
    até `tracking_max_jump` (0,15 em coordenadas normalizadas) do punho anterior.
    É uma histerese espacial: duas mãos de tamanho parecido não alternam.
  - `HAND_SELECTION_CONFIG` em `config/settings.py`: `max_num_hands=2`,
    estratégia `largest` por padrão, sobrescrevível por `LIBRIA_HAND_STRATEGY` e
    `LIBRIA_DOMINANT_HAND`.
  - Usam o seletor: classificadores estático, LSTM e híbrido, coleta
    (`scripts/collect_dataset.py`, incluindo o backfill de frames e a tela de
    espera da sequência) e ingestão de vídeos (um seletor por vídeo).
  - O módulo fica em `utils/` e não em `src/inference/` porque importar
    `src.inference` carrega o TensorFlow; a coleta e a ingestão não precisam dele.
- **Correção junto**: a coleta gravava em `capture_hand` o rótulo bruto do
  MediaPipe. Esse rótulo supõe imagem espelhada (selfie), e o LibrIA não espelha
  os quadros, então a mão registrada saía **invertida**. Agora grava a mão real
  da pessoa. Amostras coletadas antes desta data podem ter `capture_hand`
  trocado; como a coleta da Fase 1 ainda não foi feita, o impacto esperado é
  pequeno, mas confira antes de usar esse campo em análise.
- **Verificação**: 23 testes em `tests/test_hand_selection.py`, com resultados
  falsos no formato do MediaPipe: estratégias, inversão de lateralidade,
  rastreamento, perda e reaquisição da mão, e os pontos de entrada da coleta e da
  ingestão. Suíte completa: 200 passando, 2 pulados.
- **Limites**: **não testado com webcam nem com MediaPipe real** (indisponível
  neste ambiente). Os loops de câmera dos classificadores LSTM e híbrido não têm
  teste automatizado. `max_num_hands=2` tem custo de processamento maior que 1,
  não medido. A estratégia padrão `largest` é uma heurística (a mão que sinaliza
  costuma estar mais perto da câmera); `dominant` é mais confiável quando a mão
  dominante é conhecida.
- **Pendente** (continua no ROADMAP): sinais feitos com as duas mãos, que exigem
  mudar o contrato de features dos modelos (2 × 63) e retreinar.

### H-000 — Itens concluídos antes deste histórico (migrados do ROADMAP)
- **Data da migração**: 2026-11-03 · **Origem**: seções "Concluído" do `ROADMAP.md` (última atualização 2026-11-01)
- Movidos sem alteração de conteúdo, para o roadmap listar só o que falta.
  Detalhes por mudança estão no [CHANGELOG.md](CHANGELOG.md).

**v1.0.0**
- Coleta de dados de gestos de Libras
- Processamento e normalização de landmarks (`wrist_relative` e `bounding_box`)
- Modelo Random Forest (`src/model_training/libras_model_trainer.py`; a acurácia de 99% citada numa versão anterior do roadmap não foi reverificada)
- Inferência em tempo real via webcam
- Documentação básica
- Testes unitários

**Depois da v1.0.0**
- Camada 1 de robustez: features `wrist_relative` (padrão) e `bounding_box` (legado)
- Camada 2 de robustez: calibração de câmera (`scripts/calibrate_camera.py`)
- Espelhamento esquerda/direita em datasets estáticos e temporais, com backfill de `frame_XXX.png` legados
- Pipeline temporal robusto sobre o modelo LSTM (`src/inference/temporal_pipeline.py`)
- Código de treino das CNNs embarcadas e do exportador de bundle (`src/model_training/libras_embedded_*`). A exportação TFLite int8 foi corrigida (via SavedModel, `src/model_training/tflite_export.py`) e coberta por teste; ainda não foi rodada com o dataset real
- Cobertura de dataset e relatório de lacunas (`src/dataset/coverage.py`, `make report`)
- Ingestão de datasets externos em vídeo (`src/dataset/video_ingest.py`, `make ingest`)
- Metadados de proveniência por amostra (`src/dataset/sample_metadata.py`)
- Divisão de dados por pessoa, para evitar vazamento entre treino/validação/teste (`src/evaluation/dataset_splits.py`)

**Segurança de dependências (parte já feita)**
- 106 → 1 advisory na árvore completa (ver `docs/SEGURANCA.md`)
- `requirements-gpu.txt` alinhado ao TensorFlow da época (`tensorflow[and-cuda]==2.16.1`; o torch 2.13 já traz o próprio CUDA). Depois, o commit `01b08f4` passou os dois requirements para 2.19.1 (ver H-003). Não testado em GPU
- PR #18 do Dependabot fechado sem merge (propunha uma combinação torch/torchvision insolúvel)
- numpy de volta a 1.26.4 (o bump para 2.5.4 no #22 era incompatível com o TensorFlow da época); o Dependabot ignora majors do numpy
