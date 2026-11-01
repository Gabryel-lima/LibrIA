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
- **Testes**: `pytest tests` passa com 174 testes em Python 3.11 e 3.12 (TensorFlow 2.16.1 real, Keras 3.15.0, Pillow 13.0.0). Esses ambientes **não** tinham o `torch` instalado.
- **Limite importante**: nenhum teste do projeto importa TensorFlow, Keras, Pillow ou torch. Os 174 testes passariam mesmo se essas bibliotecas estivessem quebradas. A validação das subidas foi feita com scripts de fumaça, não com a suíte.
- **Keras 3.13.2 → 3.15.0**: `model/libras_lstm.keras` carrega nas duas versões e produz a mesma saída para a mesma entrada. O treino de um modelo pequeno (`fit`) também roda.
- **torch 2.13.0 + torchvision 0.28.0 + Pillow 13.0.0** (instalados, com `pip check` limpo): `import`, `transforms.ToPILImage()`, um pipeline `ToPILImage → Resize → ToTensor` e um forward de `nn.Linear` funcionam.
- **Não verificado — código que usa torch**: `src/models/transformer-gpt/` (experimental) não pôde ser importado fora do seu diretório (`No module named 'models'`), e `src/ASLnet.py` e `src/cam.py` já estavam quebrados (ver "Problemas encontrados"). O efeito da subida do torch (2.5 → 2.13, versão major) nesse código **não foi testado**.
- **Pillow** é usado diretamente só em `src/ASLnet.py` e, via torchvision, em `src/cam.py` (`ToPILImage`). Ambos são código legado, fora do fluxo principal.
- **python-dotenv**: não importado em nenhum módulo do projeto.
- **CI**: os passos de lint (`pylint`, `black`, `mypy`) têm `continue-on-error` e `--exit-zero`, então a subida dos linters não derruba o CI.

## Risco residual

### protobuf 4.25.9 — PYSEC-2026-1805 (CVE-2026-0994, GHSA-7gcm-g887-7qv7)
- **O quê**: DoS em `google.protobuf.json_format.ParseDict()` com `Any` aninhado, porque o limite de profundidade de recursão pode ser contornado.
- **Por que não foi corrigido**: a correção está em protobuf 5.29.6 ou 6.33.5, mas `tensorflow==2.16.1` exige `protobuf<5` e `mediapipe==0.10.14` exige `protobuf<5`. Subir o protobuf exige subir TensorFlow e MediaPipe, o que é uma migração própria.
- **Exposição real**: baixa. O projeto não chama `json_format` e não parseia protobuf vindo de fonte não confiável. Ainda assim, é um risco aberto.
- **Caminho**: avaliar a migração para uma versão do TensorFlow e do MediaPipe que aceite protobuf 5.x.

## Pontos de atenção da mudança do torch

- **O PR #18 do Dependabot não pode ser mergeado como está.** Ele propõe `torch>=2.13.0,<2.14` junto com `torchvision>=0.20.1,<0.21`, combinação **insolúvel** (confirmado com `uv pip compile`): o `pip install -r requirements-dev.txt` do CI falharia. Ele também propõe `Pillow==12.3.0`, que deixa de fora os advisories corrigidos só em 13.0.0, e `notebook==7.5.6`, que ainda tem um advisory. As mudanças deste branch cobrem o #18 e vão conflitar com ele nos mesmos arquivos. Decisão do mantenedor: fechar o #18 depois do merge.
- **`requirements-gpu.txt`** fixa as bibliotecas NVIDIA do CUDA 12.4 (as do `torch 2.5.1`). O `torch 2.13.0` instalado pelo PyPI traz as próprias bibliotecas CUDA 13 (`nvidia-cudnn-cu13` etc.). O arquivo audita limpo, mas ficou **desalinhado** e provavelmente redundante. Decidir se atualiza ou remove. Não alterado neste branch.
- **Requisitos de driver NVIDIA** do torch 2.13 (CUDA 13): não verificados.

## Problemas encontrados que não são de dependências

Estes não foram causados pelas atualizações. Foram verificados no baseline (antes das mudanças).

1. **Exportação TFLite int8 quebra** (`LibrasEmbeddedCNNTrainer._export_tflite_int8` e a versão temporal). `tf.lite.TFLiteConverter.from_keras_model` aborta o processo com `LLVM ERROR: Failed to infer result type(s)`, mesmo sem quantização, com Keras 3.13.2 e 3.15.0. A exportação via SavedModel funciona e gera o int8 corretamente, verificada com as duas versões do Keras (`model.export(dir)` + `TFLiteConverter.from_saved_model(dir)`). Isso bloqueia `make embedded-train` e, por consequência, o app Android previsto em `.github/plans/android_plan.md`.
2. **`model/best_temporal_model.h5` não carrega** com Keras 3.13.2 nem 3.15.0: `Unrecognized keyword arguments passed to LSTM: {'time_major': False}`. Não sei se o arquivo é usado; o caminho configurado é `model/libras_lstm.keras`.
3. **`src/ASLnet.py` e `src/cam.py` não importam**: `src/ASLnet.py` importa `src.backup_tests.camBackup`, módulo que não existe no repositório (foi removido no commit "remove unused test files"), e ambos usam `from conf import ...`, que só funciona com a pasta `src/` no caminho.

## Como reauditar

```bash
# Resolve a árvore completa (inclui torch e transitivas) e audita
uv pip compile -p 3.11 requirements-dev.txt -o /tmp/resolved.txt
pip-audit -r /tmp/resolved.txt --no-deps --disable-pip
```
