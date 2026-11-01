# Plano — App Android para testar o modelo no celular

Status: **planejado, não iniciado**. Este documento só define o plano. Nenhum código de app foi escrito.
Criado em 2026-11-01.

## 1. Objetivo e escopo

Criar um app Android que captura a câmera, detecta a mão e roda o modelo de reconhecimento de Libras **no próprio celular**, mostrando a letra ou sinal reconhecido. O objetivo é **testar o modelo em condições reais**, não lançar um produto.

Dentro do escopo:
- Reconhecimento estático (alfabeto, pelo modelo CNN embarcado).
- Reconhecimento temporal para J e Z (pelo modelo temporal embarcado).
- Overlay com landmarks, letra detectada, confiança e FPS.
- Registro local dos resultados para comparação posterior com o host.

Fora do escopo agora:
- Raspberry Pi / Pico (fica para depois, reutilizando o bundle e o runtime C++ já existentes).
- Vocabulário lexical e funcional (depende da coleta da FASE 1).
- Publicação na Play Store.

## 2. Estado atual verificado (base para o plano)

| Item | Situação | Consequência |
|---|---|---|
| Modelo estático de produção | Random Forest (`model/model.pickle`) | Não roda direto no Android. Não será portado. |
| Modelo estático embarcado | CNN int8 (`make embedded-train` gera `model/libria_embedded_cnn_int8.tflite`) | **Usar este.** Entrada `(21, 3)`, `wrist_relative`. |
| Modelo temporal embarcado | CNN int8 (`libria_embedded_temporal_cnn_int8.tflite`), entrada `(30, 63)` | Usar este. Só classes **J e Z** (`allowed_classes`). |
| Arquivos `.tflite` no repositório | **Ausentes**. Só existem `.h5`, `.keras` e `.pickle` | Precisam ser gerados localmente, com TensorFlow instalado e o dataset local. |
| Dataset de treino | `dataset/static` e `dataset/temporal` não estão versionados | Sem o dataset local, não há como gerar os `.tflite`. |
| Métricas dos modelos embarcados | Não encontradas no repositório | Não sabemos a acurácia do modelo que vai para o celular. Medir antes de confiar. |
| Runtime embarcado em Python | `src/inference/libras_embedded_runtime.py` faz a inferência com TFLite, mas **não tem** a máquina de estados temporal | A segmentação de sinais (`TemporalPipeline`) precisa ser portada. |
| Runtime C++ (`src/interfaces/`) | Stubs com `TODO` | Não é usado no Android. Serve de base para o Pico depois. |
| Ambiente de build deste container | Java e Gradle presentes; **sem Android SDK** | APK não pode ser compilado aqui. Build na máquina do desenvolvedor ou em CI. |

## 3. Contrato que o app precisa reproduzir

O app é um **segundo consumidor** do pipeline Python. Se ele divergir, o teste no celular não diz nada sobre o modelo. Os valores abaixo vêm do código atual.

### 3.1 Landmarks → features (`utils/helpers.py`, `landmarks_to_relative`)

1. Obter 21 landmarks `(x, y, z)` da mão, com MediaPipe Hands.
2. Subtrair o pulso (landmark 0) de todos os pontos.
3. Dividir pelo maior módulo do vetor (`max_distance`), se for > 0.
4. Achatar em vetor de 63 floats.

Hipótese a verificar: o MediaPipe Android (Hand Landmarker) devolve `z` na mesma escala que o Python. Não está garantido. A paridade numérica (fase F1) é quem confirma.

### 3.2 Entrada do modelo estático (`libras_embedded_runtime.py`)

- Tensor `(21, 3)` em float32, depois **quantizado** para int8 com a fórmula do wrapper:
  `q = round(x / scale + zero_point)`, com clip ao intervalo do dtype.
- Saída: float int8, desquantizada com `(q - zero_point) * scale` e normalizada para soma 1.

No Android, usar a API do TFLite com a mesma fórmula. Não confiar na conversão automática do interpreter sem testar.

### 3.3 Entrada do modelo temporal

- Janela de `sequence_length = 30` quadros, cada um com 63 features `wrist_relative`.

### 3.4 Regras de decisão

- Limiar estático: `0.75` (`INFERENCE_CONFIG['static_confidence_threshold']`).
- Limiar temporal para J/Z: `0.85` (`LSTM_CONFIG['confidence_threshold']`).
- Regra híbrida (`choose_embedded_prediction`): se o temporal prevê J ou Z com confiança ≥ 0.85, ele vence. Senão, vale o estático com confiança ≥ 0.75. Senão, nada.

### 3.5 Segmentação temporal (`src/inference/motion_detector.py` e `temporal_pipeline.py`)

Parâmetros em `TEMPORAL_PIPELINE_CONFIG` (`config/settings.py`):
- Energia de movimento = distância euclidiana média entre landmarks de quadros consecutivos, suavizada por média móvel exponencial (`motion_smoothing = 0.6`).
- Início de sinal com energia ≥ 0.012; fim com energia ≤ 0.006 (histerese).
- `min_start_frames=2`, `min_end_frames=5`, `min_segment_frames=8`, `min_duration_seconds=0.2`, `max_duration_seconds=4.0`, `max_absent_frames=5`.
- Suavização de janela 5 e supressão de duplicatas em 1.0 s.
- Enquanto a mão se move (sinal temporal em curso), o estático fica silenciado; cooldown de 0.5 s após um sinal temporal.

**Decisão**: portar esta máquina de estados para Kotlin, com os mesmos nomes de parâmetros. Não reinventar. A validação é por fixtures (fase F1).

## 4. Decisões de arquitetura (com recomendação)

| Decisão | Opções | Recomendação | Motivo |
|---|---|---|---|
| Qual modelo estático no celular | (a) CNN int8 embarcada; (b) portar o Random Forest; (c) converter o LSTM | **(a)** | É o que o bundle já exporta. O RF não tem caminho limpo para Android. |
| Detector de mão | (a) MediaPipe Hands/Hand Landmarker Android; (b) detector próprio | **(a)** | Mesmo algoritmo do host, o que facilita a paridade. |
| Linguagem e UI | Kotlin + Jetpack Compose + CameraX; ou Flutter; ou React Native | **Kotlin + Compose + CameraX** | Melhor suporte oficial a MediaPipe e TFLite. Menos camadas para depurar. |
| Onde mora a lógica de features e segmentação | No app (Kotlin) | **No app**, validada por fixtures | Latência e funcionamento offline. |
| Fonte de verdade dos parâmetros | Hardcoded no app; ou lido do `embedded_bundle.json` | **Ler do manifesto** sempre que possível | Evita divergência quando o modelo for retreinado. |

## 5. Fases

### F0 — Pré-requisitos (bloqueante)
- Ter o dataset local em `dataset/static/` e `dataset/temporal/`.
- Ter TensorFlow instalado (o `requirements-dev.txt` pina `tensorflow==2.16.1`, que não tem wheel para Python 3.13; usar Python 3.11 ou 3.12).
- Rodar `make embedded-train` e `make embedded-export`. Isso gera os `.tflite` e o `embedded_bundle.json`.
- Rodar `make embedded-check`. Se falhar, o bundle não serve. Parar aqui.
- Registrar as métricas dos dois modelos embarcados. Sem isso, não há o que testar no celular.

**Critério de saída:** `embedded-check` passa; métricas por classe registradas.

### F1 — Fixtures de paridade (Python → Android)
- Script no host que gera um JSON com: sequências de landmarks de exemplo, features `wrist_relative` esperadas, probabilidades esperadas dos dois modelos e o token final.
- O app terá testes unitários que consomem esses fixtures.
- Tolerância numérica definida antes (ex.: diferença máxima de 1e-4 nas features, e igualdade do token final).

**Critério de saída:** testes unitários de features e de classificação passam com os fixtures.

### F2 — Esqueleto do app
- Projeto Android com CameraX, MediaPipe Hand Landmarker e overlay dos 21 pontos.
- Sem inferência ainda. Só mostra a mão detectada e o FPS.

**Critério de saída:** roda em um celular real, detecta a mão a pelo menos 15 FPS. (Número a confirmar; é uma meta inicial, não medida.)

### F3 — Inferência estática no celular
- Carregar o `.tflite` estático, aplicar a quantização e mostrar a letra com confiança.
- Respeitar o limiar de 0.75.

**Critério de saída:** para as letras do dataset, o app reproduz as probabilidades dos fixtures dentro da tolerância.

### F4 — Pipeline temporal (J e Z)
- Portar `MotionDetector` e `TemporalPipeline` para Kotlin.
- Carregar o `.tflite` temporal e aplicar a regra híbrida da seção 3.4.

**Critério de saída:** para vídeos gravados de J e Z, o app emite os mesmos tokens que o Python no mesmo vídeo.

### F5 — Teste de campo
- Protocolo com pelo menos 3 pessoas (a mesma exigência da FASE 1), variando mão, iluminação e distância.
- Registrar cada predição com timestamp e rótulo verdadeiro, em um CSV local para análise.
- Comparar com o host sobre as mesmas gravações.

**Critério de saída:** métricas por classe e taxa de rejeição registradas, separadas por pessoa.

## 6. Riscos

| Risco | Probabilidade | Mitigação |
|---|---|---|
| `z` do MediaPipe Android diferente do Python | Média (hipótese não verificada) | Fixtures de F1 detectam logo. |
| Perda de acurácia pela quantização int8 | Média | Comparar acurácia float vs int8 no `embedded-check`. |
| Câmera frontal espelhada muda a mão dominante | Alta | Mirror augmentation já existe no treino; testar ambas as câmeras. |
| Modelo embarcado com acurácia baixa, sem métrica registrada | Desconhecida | F0 exige métricas antes de F3. |
| Dataset local ausente no seu ambiente | Desconhecida | Confirmar antes de começar. |
| Latência alta em celulares antigos | Média | Medir em pelo menos 2 aparelhos. |

## 7. Ambiente de build

- Container atual: Java e Gradle existem, mas **sem Android SDK**. Não compila APK aqui.
- Opções: (a) Android Studio na sua máquina; (b) CI no GitHub Actions com `setup-android`. A opção (a) é mais rápida para iterar; a (b) garante que o build não depende de uma máquina específica.

## 8. Critério de sucesso geral

- Paridade com o host dentro da tolerância definida em F1 e F4.
- Métricas por classe registradas em teste de campo com pelo menos 3 pessoas.
- Build reproduzível por qualquer pessoa com o procedimento documentado.

## 9. Próximos passos

1. Confirmar que o dataset local existe e onde está (F0).
2. Escolher a opção de build (seção 7).
3. Gerar o bundle e rodar `embedded-check` (F0).
4. Só então começar F1.

Perguntas em aberto:
- O dataset local tem `dataset/static` e `dataset/temporal` populados?
- Você tem um Android físico para testes? Qual modelo e versão?
- Prefere Android Studio na sua máquina ou CI?
