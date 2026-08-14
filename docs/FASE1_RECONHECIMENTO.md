# Fase 1 — Consolidar o reconhecimento de sinais

Referência: [`.github/plans/atualização_projeto.md`](../.github/plans/atualização_projeto.md).

Esta fase define **o que** o sistema reconhece e **como** medimos isso. A
infraestrutura de código está pronta; a coleta com múltiplas pessoas é o
trabalho de campo que roda em cima dela.

## 1. Vocabulário

Fonte única de verdade: [`config/vocabulary.py`](../config/vocabulary.py).
`config/settings.py` deriva suas listas de labels dali — nunca duplique labels.

| Família | Modalidade | Conteúdo |
|:--------|:-----------|:---------|
| `alphabet` | `static` (24) + `temporal` (J, Z) | Alfabeto manual |
| `lexical` | `temporal` | Palavras completas (OI, OBRIGADO, AJUDA, …) |
| `functional` | `temporal` | ESPACO, PAUSA, APAGAR, CONFIRMAR |

Além das três famílias existe `UNKNOWN_LABEL = 'DESCONHECIDO'`: a classe
explícita de **fora do vocabulário**. Ela não é um sinal, é o destino de tudo
que o modelo não deve afirmar — tanto amostras negativas coletadas de propósito
quanto predições abaixo do limiar de confiança.

Para ampliar o vocabulário, acrescente entradas em `_LEXICAL_ENTRIES`. Labels
precisam ser ASCII, maiúsculas e sem espaço (viram nome de diretório).

## 2. Metadados por amostra

Cada `.npy` ganha um `.json` irmão com o mesmo nome
([`src/dataset/sample_metadata.py`](../src/dataset/sample_metadata.py)). Datasets
antigos continuam válidos: quem só lê landmarks ignora os `.json`.

Campos: `label`, `modality`, `sign_type`, `subject_id`, `camera_id`,
`environment`, `dominant_hand`, `capture_hand`, `duration_seconds`, `quality`,
`feature_mode`, `feature_dimension`, `sequence_length`, `mirrored`,
`source_sample`, `created_at`, `schema_version`.

A amostra espelhada herda os metadados da original com `mirrored: true` e a
lateralidade invertida.

### Coletar com metadados

```bash
# Alfabeto estático
python -m scripts.collect_dataset static \
  --subject pessoa_01 --camera-id webcam_c920 \
  --environment sala_luz_natural --dominant-hand right

# Palavras completas (vocabulário lexical)
python -m scripts.collect_dataset temporal --vocabulary lexical \
  --subject pessoa_01 --camera-id webcam_c920 \
  --environment sala_luz_natural --dominant-hand right

# Amostras negativas (fora do vocabulário)
python -m scripts.collect_dataset temporal --vocabulary unknown --subject pessoa_01
```

Ou via Make:

```bash
make collect-temporal CAPTURE_SUBJECT=pessoa_01 CAPTURE_ENVIRONMENT=sala \
  CAPTURE_CAMERA_ID=webcam_c920 CAPTURE_DOMINANT_HAND=right
```

**Sem `--subject` não há divisão por pessoa possível.** A coleta avisa quando o
campo fica no padrão.

## 3. Divisão treino/validação/teste por pessoa

[`src/evaluation/dataset_splits.py`](../src/evaluation/dataset_splits.py).
A unidade de divisão é o `subject_id`, não a amostra: dividir aleatoriamente
faz a mesma pessoa aparecer nos três conjuntos e o modelo aprende a pessoa, não
o sinal.

```python
from src.evaluation.dataset_splits import assert_no_subject_leakage, split_metadata_by_person
from src.dataset.sample_metadata import collect_dataset_metadata

metadata = collect_dataset_metadata('dataset/temporal')
split = split_metadata_by_person(metadata)   # 60/20/20 por padrão
assert_no_subject_leakage(split)
```

A atribuição é determinística e gulosa. Com menos pessoas do que conjuntos, a
função **falha** em vez de devolver um teste vazio — passe
`allow_empty_splits=True` só se souber o que está abrindo mão.

## 4. Métricas

[`src/evaluation/metrics.py`](../src/evaluation/metrics.py). Além de precisão,
recall, F1 e matriz de confusão por classe:

* **taxa de rejeição por classe** — quanto de cada classe cai abaixo do limiar
  e vira `DESCONHECIDO`;
* **latência** — média, p50, p95 e máximo, para validar tempo real.

```python
from src.evaluation.metrics import evaluate_predictions, format_report

report = evaluate_predictions(
    y_true, y_pred,
    confidences=confidences,
    rejection_threshold=0.75,   # EVALUATION_CONFIG['rejection_threshold']
    latencies_ms=latencies,
)
print(format_report(report))
```

Uma predição rejeitada não conta como acerto nem como erro de outra classe: vai
para a coluna `DESCONHECIDO`. As métricas macro ignoram essa coluna e a
rejeição é reportada à parte — recusar é melhor que traduzir errado com ar de
certeza.

## 5. Relatório de cobertura

```bash
make dataset-report          # ou: python -m scripts.dataset_report --json out.json
```

Mostra, por modalidade: total de amostras, cobertura de metadados, pessoas,
ambientes e câmeras presentes, labels do vocabulário ainda sem dados e a
divisão por pessoa.

## 6. O que falta para fechar a fase

O código está pronto; o que resta é trabalho de coleta:

- [ ] Coletar as palavras de `_LEXICAL_ENTRIES` e os gestos funcionais.
- [ ] Coletar amostras de `DESCONHECIDO` (gestos fora do vocabulário).
- [ ] Coletar ao menos 3 pessoas — mínimo para treino/validação/teste sem vazamento.
- [ ] Variar mão, velocidade, iluminação, distância e câmera.
- [ ] Rodar `make dataset-report` e conferir cobertura por classe e por pessoa.
- [ ] Anotar metadados dos dados legados (hoje 0% de cobertura em `dataset/static`).

Para treinar com o vocabulário ampliado, aponte
`LSTM_CONFIG['allowed_classes']` para `TEMPORAL_VOCABULARY_LABELS` e deixe
`require_all_allowed_classes = False`, para treinar com as classes já coletadas
em vez de falhar nas que ainda faltam.
