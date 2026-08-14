# Fase 2 — Reconhecimento temporal robusto

Referência: [`.github/plans/atualização_projeto.md`](../.github/plans/atualização_projeto.md).
Fase anterior: [FASE1_RECONHECIMENTO.md](FASE1_RECONHECIMENTO.md).

## O que mudou

Antes, o reconhecimento temporal era uma janela fixa de 30 quadros: o LSTM era
consultado a cada quadro assim que o buffer enchia, houvesse ou não um sinal
sendo executado. O modelo estático rodava em paralelo a cada N quadros e um
árbitro decidia entre os dois.

Agora existe um pipeline explícito:

```
features → buffer → movimento → segmentação → modelo → suavização
         → supressão de duplicatas → SignToken
```

O modelo temporal só é consultado quando há um sinal **delimitado por
movimento**, e o modelo estático responde apenas enquanto a mão está **parada**.

## Componentes

| Módulo | Responsabilidade |
|:-------|:-----------------|
| [`temporal_buffer.py`](../src/inference/temporal_buffer.py) | Histórico com carimbo de tempo e reamostragem para `sequence_length` |
| [`motion_detector.py`](../src/inference/motion_detector.py) | Energia de movimento (deslocamento médio por landmark, suavizado) |
| [`sign_segmenter.py`](../src/inference/sign_segmenter.py) | Máquina de estados de início/fim com histerese |
| [`prediction_smoother.py`](../src/inference/prediction_smoother.py) | Média móvel de probabilidades e supressão de duplicatas |
| [`sign_token.py`](../src/inference/sign_token.py) | Saída padronizada |
| [`temporal_pipeline.py`](../src/inference/temporal_pipeline.py) | Orquestra tudo, com fallback estático |

### Histerese na segmentação

Começar um sinal exige energia acima de `motion_start_threshold`; encerrá-lo
exige energia abaixo de `motion_end_threshold`, que é **menor**. Sem essa
diferença, uma pausa breve no meio de um sinal o cortaria em dois. Os quadros
parados do final são removidos do segmento antes da classificação.

### Duração variável

Como o segmento tem o tamanho que o gesto teve, ele é reamostrado para
`sequence_length` só na hora de inferir. Isso é o que permite treinar palavras
com durações diferentes sem forçar todo mundo a 30 quadros na captura.

## Saída padronizada

Todo reconhecimento sai como `SignToken`:

```python
{
  'label': 'OI', 'token': 'oi', 'confidence': 0.91,
  'start_time': 12.4, 'end_time': 13.1, 'duration_seconds': 0.7,
  'source': 'temporal',      # temporal | static
  'state': 'final',          # partial | final | rejected
  'sign_type': 'lexical',    # alphabet | lexical | functional
  'frame_count': 21,
}
```

`state` é o que evita apresentar tradução errada como certeza:

* `partial` — hipótese enquanto o sinal ainda está em execução (feedback de UI);
* `final` — sinal encerrado e acima do limiar de confiança;
* `rejected` — sinal encerrado abaixo do limiar; vira `DESCONHECIDO`, com
  `token` vazio.

Esse é o contrato de entrada da camada de composição linguística (Fases 3 e 4).

## Fallback estático

O modelo estático continua respondendo por letras, mas com três guardas que
antes não existiam ou estavam espalhadas:

1. só opina com leitura de movimento válida (dois quadros consecutivos com mão);
2. só opina com energia abaixo de `motion_end_threshold` (mão realmente parada);
3. fica em silêncio por `static_cooldown_seconds` após um sinal temporal — o
   retorno da mão ao repouso passa por poses que seriam lidas como letras.

A guarda 3 preserva o comportamento que o `PredictionMerger` tinha via
`prediction_cooldown_seconds`.

## Configuração

Tudo em `TEMPORAL_PIPELINE_CONFIG` ([config/settings.py](../config/settings.py)).
Os limiares de movimento estão em unidades de deslocamento médio de landmark
por quadro e **precisam ser calibrados na sua câmera** — o padrão
(`0.012` / `0.006`) é um ponto de partida, não uma medição.

Para calibrar, imprima `pipeline.last_energy` com a mão parada e depois
sinalizando: `motion_start_threshold` deve ficar acima do ruído de repouso e
`motion_end_threshold` entre os dois.

## Uso

O pipeline recebe os modelos por injeção de dependência, o que permite trocar
LSTM por CNN temporal ou Transformer sem tocar nele:

```python
from src.inference import TemporalPipeline

pipeline = TemporalPipeline(
    temporal_predictor=lambda sequence: model.predict(sequence[None])[0],
    label_map={0: 'J', 1: 'Z'},
    sequence_length=30,
    static_predictor=lambda features: ('A', 0.93),
)

token = pipeline.process_frame(features, timestamp, hand_present=True)
if token is not None and token.is_final:
    print(token.to_dict())
```

`LibrasHybridRealtimeClassifier` já usa esse caminho — o loop de câmera agora
apenas extrai landmarks e entrega ao pipeline.

## O que ficou de fora desta fase, e por quê

Dois itens do plano dependem de decisões ou dados que ainda não existem:

* **Comparar LSTM, CNN temporal e Transformer no mesmo protocolo.** O protocolo
  de avaliação existe (Fase 1: split por pessoa + métricas por classe), mas o
  dataset temporal tem só `J` e `Z`, de uma pessoa. Comparar arquiteturas nesses
  dados mediria ruído, não qualidade.
* **Duas mãos, pose corporal e expressões faciais.** Isso muda
  `FEATURE_DIMENSION` de 63 para algo bem maior e invalida todo o dataset já
  coletado, os modelos treinados e o bundle embarcado — inclusive o contrato do
  runtime C++/Pico. É uma decisão arquitetural que vale ser tomada
  explicitamente, e de preferência junto com a expansão do vocabulário, para
  recoletar uma vez só.

O pipeline não impede nenhum dos dois: os modelos entram por injeção, e o
tamanho do vetor de features é lido do modelo, não fixado no código.
