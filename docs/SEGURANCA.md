# Segurança de dependências

Registro das vulnerabilidades de dependências encontradas em 2026-11-01, das correções aplicadas e dos riscos que continuam abertos.

## Como foi medido

Auditoria com `pip-audit` (base OSV/PyPI, consultada em 2026-11-01) sobre a **árvore completa resolvida**, incluindo `torch` e todas as dependências transitivas:

1. `uv pip compile requirements-dev.txt` em Python 3.11 e 3.12 gera o conjunto exato de versões que o CI instalaria (230 pacotes no `main` original, 238 depois das mudanças).
2. `pip-audit -r <conjunto resolvido> --no-deps --disable-pip` audita cada versão resolvida.

A contagem do GitHub Dependabot (72 alertas no branch padrão, em 2026-11-01) não foi reconciliada com esta auditoria: o acesso a alertas do Dependabot não estava disponível no ambiente de trabalho.

> **Correção de uma versão anterior deste documento:** ela reportava "87 → 1". Esse "87" excluía o `torch`, que o `pip-audit` não consegue auditar quando o requirement é uma faixa (`torch>=2.5.1,<2.9`). A árvore completa do `main` tinha **106** advisories, dos quais 19 no `torch 2.5.1`.

## Antes e depois

| Pacote | Antes | Depois | Advisories corrigidos | Escopo |
|---|---|---|---|---|
| torch | 2.5.1 (resolvido da faixa `>=2.5.1,<2.9`) | **2.13.0** | 19 | produção |
| torchvision | 0.20.1 | **0.28.0** | (par obrigatório do torch) | produção |
| keras | 3.13.2 | **3.15.0** | 8 | produção |
| pillow | 12.1.1 | **13.0.0** | 57 | produção |
| python-dotenv | 1.0.1 | **1.2.4** | 1 | produção |
| pytest | 7.4.3 | **9.0.3** | 1 | desenvolvimento |
| black | 24.3.0 | **26.10.1** | 3 | desenvolvimento |
| pylint | 3.0.3 | **4.0.9** | 1 | desenvolvimento |
| notebook | 7.2.2 | **7.6.3** | 3 | desenvolvimento |
| jupyterlab | 4.2.7 (transitiva) | **4.6.4** (fixada) | 12 | desenvolvimento |

Resultado: de **106 advisories únicos em 10 pacotes** para **1 advisory em 1 pacote** (ver risco residual). O resultado é o mesmo em Python 3.11 e 3.12.

Por que o par `torch` + `torchvision` mudou junto: o `torchvision 0.20.x` exige `torch 2.5.x`, então as faixas antigas travavam o torch na versão vulnerável. Um par consistente é `torch 2.13.x` + `torchvision 0.28.x` (resolvido e verificado).

Mudanças no repositório:
- `requirements.txt`: torch, torchvision, keras, pillow e python-dotenv atualizados; comentário sobre o protobuf.
- `requirements-dev.txt`: pytest, black, pylint, notebook atualizados; `jupyterlab==4.6.4` fixado explicitamente.
- `.github/dependabot.yml`: o arquivo tinha `package-ecosystem: ""`, o que é inválido e fazia o Dependabot não abrir nenhum PR. Agora cobre `pip` e `github-actions`.

## Validação

- **Resolução**: a árvore completa de `requirements-dev.txt` resolve em Python 3.11 e 3.12.
- **Testes**: `pytest tests` passa com 176 testes em Python 3.11 e 3.12 (TensorFlow 2.16.1 real, Keras 3.15.0, Pillow 13.0.0); sem TensorFlow, 2 deles são pulados. Esses ambientes **não** tinham o `torch` instalado.
- **Limite importante**: quase nenhum teste do projeto exercita TensorFlow, Keras, Pillow ou torch. A exceção é `tests/test_tflite_export.py` (exportação TFLite int8, em subprocesso). Os demais testes passariam mesmo com essas bibliotecas quebradas. A validação das subidas foi feita com scripts de fumaça.
- **Keras 3.13.2 → 3.15.0**: `model/libras_lstm.keras` carrega nas duas versões e produz a mesma saída para a mesma entrada. O treino de um modelo pequeno (`fit`) também roda.
- **torch 2.13.0 + torchvision 0.28.0 + Pillow 13.0.0** (instalados, com `pip check` limpo): `import`, `transforms.ToPILImage()`, um pipeline `ToPILImage → Resize → ToTensor` e um forward de `nn.Linear` funcionam.
- **Não verificado — código que usa torch**: `src/models/transformer-gpt/` (experimental) não pôde ser importado fora do seu diretório (`No module named 'models'`), e `src/ASLnet.py` e `src/cam.py` já estavam quebrados (ver "Problemas encontrados"). O efeito da subida do torch (2.5 → 2.13, versão major) nesse código **não foi testado**.
- **Pillow** é usado diretamente só em `src/ASLnet.py` e, via torchvision, em `src/cam.py` (`ToPILImage`). Ambos são código legado, fora do fluxo principal.
- **python-dotenv**: não importado em nenhum módulo do projeto.
- **CI**: os passos de lint (`pylint`, `black`, `mypy`) têm `continue-on-error` e `--exit-zero`, então a subida dos linters não derruba o CI.

## Risco residual

### protobuf 4.25.9 — PYSEC-2026-1805 (CVE-2026-0994, GHSA-7gcm-g887-7qv7)
- **O quê**: DoS em `google.protobuf.json_format.ParseDict()` com `Any` aninhado, porque o limite de profundidade de recursão pode ser contornado.
- **Por que não foi corrigido**: a correção está em protobuf 5.29.6 ou 6.33.5, mas `mediapipe==0.10.14` exige `protobuf>=4.25.3,<5`. Quando esta seção foi escrita, `tensorflow==2.16.1` também exigia `protobuf<5`.
- **Atualização (2026-11-03)**: o commit `01b08f4` fixou `tensorflow==2.19.1`, que aceita `protobuf>=3.20.3,<6` (conferido nos metadados do PyPI). **O único bloqueio agora é o MediaPipe**, que precisa da API legada `mp.solutions`, removida nas versões >=0.10.31. Ver `HISTORICO.md`, H-003.
- **Exposição real**: baixa. O projeto não chama `json_format` e não parseia protobuf vindo de fonte não confiável. Ainda assim, é um risco aberto.
- **Caminho**: migrar a captura de `mp.solutions.hands` para a API MediaPipe Tasks (o app Android já usa Tasks) e então subir o MediaPipe e o protobuf para 5.29.6+.

## Pontos de atenção da mudança do torch

- **O PR #18 do Dependabot foi fechado sem merge em 2026-11-01.** Ele propunha `torch>=2.13.0,<2.14` junto com `torchvision>=0.20.1,<0.21`, combinação **insolúvel** (confirmado com `uv pip compile`): o `pip install -r requirements-dev.txt` do CI falharia. Ele também propunha `Pillow==12.3.0`, que deixa de fora os advisories corrigidos só em 13.0.0, e `notebook==7.5.6`, que ainda tem um advisory. As mudanças deste branch cobrem o #18 e vão conflitar com ele nos mesmos arquivos. O #18 foi fechado; este branch cobre o que ele propunha.
- **`requirements-gpu.txt` — corrigido.** Fixava as bibliotecas NVIDIA do torch 2.5.1 (CUDA 12.4, cuDNN 9.1). Não serviam ao torch 2.13, que traz o próprio CUDA 13 pelo PyPI, e **conflitavam** com o TensorFlow 2.16.1, que exige cuDNN 8.9.7 (confirmado com `uv pip compile`: `tensorflow[and-cuda]==2.16.1` + `nvidia-cudnn-cu12==9.1.0.70` não tem solução). Agora o arquivo instala só `tensorflow[and-cuda]==2.16.1`; a árvore resolve em Python 3.11 e 3.12, cada biblioteca com o seu cuDNN (cu12 8.9.7 para o TensorFlow, cu13 9.20 para o torch), e o audit não muda. **Não testado em GPU** (o ambiente não tem uma).
- **Requisitos de driver NVIDIA** do torch 2.13 (CUDA 13): não verificados.
- **Incidente de 2026-11-01, consequência direta da falta de CI**: o PR #22 do Dependabot (numpy 1.26.4 → 2.5.4) foi mergeado e deixou o `requirements.txt` insolúvel, porque `tensorflow==2.16.1` exige `numpy<2`. Corrigido voltando para `numpy==1.26.4`; o `dependabot.yml` passou a ignorar majors do numpy até o TensorFlow ser migrado.
- **Sem CI por enquanto**: os workflows do GitHub Actions foram desativados (ver `CHANGELOG.md`). Nada valida automaticamente se uma atualização de dependência resolve ou se os testes passam, inclusive nos PRs do Dependabot. Até reativar, rodar à mão a resolução e a suíte (seção "Como reauditar" e `pytest tests -q`) antes de aceitar qualquer bump.

## Problemas encontrados que não são de dependências

Estes não foram causados pelas atualizações. Foram verificados no baseline (antes das mudanças).

1. **Exportação TFLite int8 abortava — corrigida.** `TFLiteConverter.from_keras_model` abortava o processo com `LLVM ERROR: Failed to infer result type(s)` (Keras 3.13.2 e 3.15.0, TensorFlow 2.16.1), mesmo sem quantização. Correção: exportar via SavedModel e converter com `from_saved_model` (`src/model_training/tflite_export.py`, usado pelos dois trainers). Verificado de ponta a ponta com dados sintéticos em Python 3.11 e 3.12: `train_model` dos dois trainers → `build_embedded_bundle` → `LibrasEmbeddedRuntime`, com o int8 concordando 100% com o Keras em float. **Limites:** os dados sintéticos são separáveis (provam a mecânica, não a qualidade do modelo) e `make embedded-train` com o dataset real não foi executado. Regressão coberta por `tests/test_tflite_export.py`, que falha no código antigo.
2. **`model/best_temporal_model.h5` e `model/temporal_cnn_model.h5` são artefatos órfãos de outro contrato de entrada.** Nenhum código os carrega. São LSTMs do Keras 2.15 (agosto de 2025) que recebem **16 quadros × 42 features** (modo `bounding_box`) e têm 26 classes; o pipeline atual usa **30 × 63** (`wrist_relative`). No Keras 3 eles só carregam removendo o argumento `time_major` do LSTM (os pesos não mudam), mas, mesmo carregados, não servem ao pipeline atual. O `temporal_model_config.pickle` que os acompanha tem metadados inconsistentes (`num_classes` 26 e 24). Mantidos no repositório por serem resultados de experimento; removê-los é decisão do mantenedor. Ver `docs/DATASETS.md`, seção 7.
3. **`src/ASLnet.py` e `src/cam.py` não importam**: `src/ASLnet.py` importa `src.backup_tests.camBackup`, módulo que não existe no repositório (foi removido no commit "remove unused test files"), e ambos usam `from conf import ...`, que só funciona com a pasta `src/` no caminho.

## Como reauditar

```bash
# Resolve a árvore completa (inclui torch e transitivas) e audita
uv pip compile -p 3.11 requirements-dev.txt -o /tmp/resolved.txt
pip-audit -r /tmp/resolved.txt --no-deps --disable-pip
```
