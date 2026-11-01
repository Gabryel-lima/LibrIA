# Segurança de dependências

Registro das vulnerabilidades de dependências encontradas em 2026-11-01, das correções aplicadas e dos riscos que continuam abertos.

## Como foi medido

Auditoria com `pip-audit` (base OSV/PyPI, consultada em 2026-11-01), em duas camadas:

1. **Pins diretos** de `requirements.txt` (sem `torch`, que é range e não pinado).
2. **Ambiente completo** congelado com `uv pip freeze` a partir de `requirements.txt` + `requirements-dev.txt`, incluindo dependências transitivas. É a camada que mais se aproxima do que o CI instala.

A contagem do GitHub Dependabot (72 alertas no branch padrão, em 2026-11-01) não foi reconciliada com esta auditoria: o acesso a alertas do Dependabot não estava disponível no ambiente de trabalho.

## Antes e depois

| Pacote | Antes | Depois | Advisories corrigidos | Escopo |
|---|---|---|---|---|
| keras | 3.13.2 | **3.15.0** | 8 | produção |
| pillow | 12.1.1 | **13.0.0** | 57 | produção |
| python-dotenv | 1.0.1 | **1.2.4** | 1 | produção |
| pytest | 7.4.3 | **9.0.3** | 1 | desenvolvimento |
| black | 24.3.0 | **26.10.1** | 3 | desenvolvimento |
| pylint | 3.0.3 | **4.0.9** | 1 | desenvolvimento |
| notebook | 7.2.2 | **7.6.3** | 3 | desenvolvimento |
| jupyterlab | 4.2.7 (transitiva) | **4.6.4** (fixada) | 12 | desenvolvimento |

Resultado do audit do ambiente completo: de **87 advisories únicos em 9 pacotes** para **1 advisory em 1 pacote** (ver risco residual).

Mudanças no repositório:
- `requirements.txt`: keras, pillow e python-dotenv atualizados; comentário sobre o protobuf.
- `requirements-dev.txt`: pytest, black, pylint, notebook atualizados; `jupyterlab==4.6.4` fixado explicitamente.
- `.github/dependabot.yml`: o arquivo tinha `package-ecosystem: ""`, o que é inválido e fazia o Dependabot não abrir nenhum PR. Agora cobre `pip` e `github-actions`.

## Validação

- **Testes**: `pytest tests` passa com 174 testes em Python 3.11 (com TensorFlow 2.16.1 real, Keras 3.15.0, Pillow 13.0.0). O ambiente 3.12 do CI foi validado separadamente.
- **Limite importante**: nenhum teste do projeto importa TensorFlow, Keras ou Pillow. Os 174 testes passariam mesmo se essas bibliotecas estivessem quebradas. A validação das subidas foi feita com scripts de fumaça, não com a suíte.
- **Keras 3.13.2 → 3.15.0**: `model/libras_lstm.keras` carrega nas duas versões e produz a mesma saída para a mesma entrada.
- **Pillow**: usado apenas em `src/ASLnet.py` (código legado, fora do fluxo principal).
- **python-dotenv**: não importado em nenhum módulo do projeto.
- **CI**: os passos de lint (`pylint`, `black`, `mypy`) têm `continue-on-error` e `--exit-zero`, então a subida dos linters não derruba o CI.

## Risco residual

### protobuf 4.25.9 — PYSEC-2026-1805 (CVE-2026-0994, GHSA-7gcm-g887-7qv7)
- **O quê**: DoS em `google.protobuf.json_format.ParseDict()` com `Any` aninhado, porque o limite de profundidade de recursão pode ser contornado.
- **Por que não foi corrigido**: a correção está em protobuf 5.29.6 ou 6.33.5, mas `tensorflow==2.16.1` exige `protobuf<5` e `mediapipe==0.10.14` exige `protobuf<5`. Subir o protobuf exige subir TensorFlow e MediaPipe, o que é uma migração própria.
- **Exposição real**: baixa. O projeto não chama `json_format` e não parseia protobuf vindo de fonte não confiável. Ainda assim, é um risco aberto.
- **Caminho**: avaliar a migração para uma versão do TensorFlow e do MediaPipe que aceite protobuf 5.x.

## Problemas encontrados que não são de dependências

Estes não foram causados pelas atualizações. Foram verificados no baseline (antes das mudanças), com as mesmas versões de Keras nas duas versões testadas.

1. **Exportação TFLite int8 quebra** (`LibrasEmbeddedCNNTrainer._export_tflite_int8` e a versão temporal). `tf.lite.TFLiteConverter.from_keras_model` aborta o processo com `LLVM ERROR: Failed to infer result type(s)`, mesmo sem quantização. A exportação via SavedModel funciona e gera o int8 corretamente (`model.export(dir)` + `TFLiteConverter.from_saved_model(dir)`). Isso bloqueia `make embedded-train` e, por consequência, o app Android previsto em `.github/plans/android_plan.md`.
2. **`model/best_temporal_model.h5` não carrega** em nenhuma das duas versões: `Unrecognized keyword arguments passed to LSTM: {'time_major': False}`. Não sei se o arquivo é usado; o caminho configurado é `model/libras_lstm.keras`.

## Como reauditar

```bash
# Auditoria dos pins diretos
pip-audit -r requirements.txt --no-deps --disable-pip

# Auditoria do ambiente completo (inclui transitivas)
uv pip freeze > /tmp/frozen.txt
pip-audit -r /tmp/frozen.txt --no-deps --disable-pip
```

Observação: o `torch` (range `>=2.5.1,<2.9`) não entra no `--no-deps` porque o `pip-audit` exige versões exatas. Auditar `torch` exige fixar uma versão primeiro.
