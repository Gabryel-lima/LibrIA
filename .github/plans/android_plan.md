# Plano — App Android para testar o modelo no celular

Status: **planejado, não iniciado**. Este documento só define o plano. Nenhum código de app foi escrito.
Criado em 2026-11-01. Atualizado em 2026-11-01: núcleo de processamento em C++ (seção 4).

## 1. Objetivo e escopo

Criar um app Android que captura a câmera, detecta a mão e roda o modelo de reconhecimento de Libras **no próprio celular**, mostrando a letra ou sinal reconhecido. O objetivo é **testar o modelo em condições reais**, não lançar um produto.

Dentro do escopo:
- Reconhecimento estático (alfabeto, pelo modelo CNN embarcado).
- Reconhecimento temporal para J e Z (pelo modelo temporal embarcado).
- Overlay com landmarks, letra detectada, confiança e FPS.
- Registro local dos resultados para comparação posterior com o host.

Fora do escopo agora:
- Raspberry Pi / Pico. Fica para depois, reutilizando o núcleo C++ (seção 4).
- Vocabulário lexical e funcional (depende da coleta da FASE 1).
- Publicação na Play Store.

## 2. Estado atual verificado (base para o plano)

| Item | Situação | Consequência |
|---|---|---|
| Modelo estático de produção | Random Forest (`model/model.pickle`) | Não roda direto no Android. Não será portado. |
| Modelo estático embarcado | CNN int8 (`make embedded-train` gera `model/libria_embedded_cnn_int8.tflite`) | **Usar este.** Entrada `(21, 3)`, `wrist_relative`. |
| Modelo temporal embarcado | CNN int8 (`libria_embedded_temporal_cnn_int8.tflite`), entrada `(30, 63)` | Usar este. Só classes **J e Z** (`allowed_classes`). |
| Arquivos `.tflite` no repositório | **Ausentes**. Só existem `.h5`, `.keras` e `.pickle` | Precisam ser gerados localmente, com TensorFlow e o dataset. A exportação int8 foi corrigida (ver `docs/SEGURANCA.md`, item 1), mas ainda não foi rodada com o dataset real. |
| Dataset de treino | `dataset/static` e `dataset/temporal` não estão versionados | Sem o dataset local, não há como gerar os `.tflite`. |
| Métricas dos modelos embarcados | Não encontradas no repositório | Não sabemos a acurácia do modelo que vai para o celular. Medir antes de confiar. |
| Runtime embarcado em Python | `src/inference/libras_embedded_runtime.py` faz a inferência com TFLite, mas **não tem** a máquina de estados temporal | A segmentação (`TemporalPipeline`) precisa ser portada. |
| Runtime C++ (`src/interfaces/`) | Stubs com `TODO` (`Init`, `PredictStatic`, `PredictTemporal`) | É o ponto de partida do núcleo C++ (seção 4). |
| Ambiente de build deste container | Java, Gradle e Python 3.11/3.12 presentes; **sem Android SDK/NDK** | APK não pode ser compilado aqui. Build na máquina do desenvolvedor ou em CI. |

## 3. Contrato que o app precisa reproduzir

O app é um **segundo consumidor** do pipeline Python. Se ele divergir, o teste no celular não diz nada sobre o modelo. Os valores abaixo vêm do código atual.

### 3.1 Landmarks → features (`utils/helpers.py`, `landmarks_to_relative`)

1. Obter 21 landmarks `(x, y, z)` da mão, com MediaPipe Hands.
2. Subtrair o pulso (landmark 0) de todos os pontos.
3. Dividir pelo maior módulo do vetor (`max_distance`), se for > 0.
4. Achatar em vetor de 63 floats.

Hipótese a verificar: o MediaPipe Android (Hand Landmarker) devolve `z` na mesma escala que o Python. Não está garantido. A paridade numérica (F1) é quem confirma.

### 3.2 Entrada do modelo estático (`libras_embedded_runtime.py`)

- Tensor `(21, 3)` em float32, depois **quantizado** para int8 com a fórmula do wrapper:
  `q = round(x / scale + zero_point)`, com clip ao intervalo do dtype.
- Saída: int8, desquantizada com `(q - zero_point) * scale` e normalizada para soma 1.

Usar a API do TFLite com a mesma fórmula. Não confiar na conversão automática do interpreter sem testar.

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

**Decisão**: portar esta máquina de estados para **C++** (núcleo, seção 4), com os mesmos nomes de parâmetros. Não reinventar. A validação é por fixtures (F1).

## 4. Arquitetura: núcleo nativo em C++, interface em Java/Kotlin

**Direção do usuário (2026-11-01):** o processamento de performance deve ficar em uma linguagem nativa, e não em Python nem diretamente em Java, com Java/Kotlin só na interface e no ciclo de vida do Android. O usuário citou C++, C e Go como candidatas e pediu a mais indicada para esse propósito.

**Recomendação (a confirmar pelo usuário): C++.** A escolha da linguagem **ainda não foi confirmada**; a seção 4.2 compara as alternativas.

### 4.1 Camadas

```
┌───────────────────────────────────────────────┐
│ UI Android (Kotlin ou Java)                    │  tela, overlay, botões, permissões
│  - CameraX entrega frames                      │  NÃO roda inferência
│  - desenha o resultado recebido                │
└───────────────┬───────────────────────────────┘
                │ JNI (uma chamada por quadro, sem alocação)
┌───────────────▼───────────────────────────────┐
│ Núcleo nativo (C++)                            │  thread de trabalho própria
│  - features wrist_relative (seção 3.1)         │
│  - MotionDetector + TemporalPipeline (3.5)     │
│  - inferência TFLite (3.2, 3.3) + regra (3.4)  │
│  - buffers pré-alocados, sem alocação por quadro│
└───────────────────────────────────────────────┘
```

Detecção de mão: MediaPipe Hand Landmarker. O núcleo do MediaPipe é C++, mas o caminho suportado no Android é a API Java/Kotlin dos Tasks. Logo, o cenário mais provável é: **a câmera e o MediaPipe ficam do lado Java**, e o núcleo C++ recebe só os 63 floats de landmarks por quadro. Chamar o MediaPipe direto do C++ é possível, mas não é o caminho simples; avaliar na F2 e medir.

### 4.2 Por que C++ e não as alternativas

| Opção | Prós | Contras | Veredito |
|---|---|---|---|
| **C++** | o TFLite tem API C++ (e C), e o núcleo do MediaPipe é C++; NDK e CMake são o caminho padrão; controle explícito de memória e de cópias; RAII sem custo; o núcleo `src/interfaces/` já é C++ | Risco de bugs de memória (mitigado com ASan/UBSan no build de host); build com NDK mais trabalhoso | **Recomendado** |
| **C puro** | Mínimo de runtime; ABI estável | Sem RAII, mais código para gerenciar memória e strings; a API C do TFLite é mais limitada que a C++ | Só se C++ causar problema concreto |
| **Rust** | Segurança de memória sem GC; bom suporte a NDK | Bindings de TFLite e MediaPipe menos maduros; mais uma linguagem no projeto | Alternativa válida; custo de integração maior |
| **Go** (`gomobile`) | Produtividade | Runtime com GC e overhead de chamada entre Go e Java a cada quadro; bindings de TFLite/MediaPipe fracos (avaliação deste plano, não medida); pausas de GC no caminho de tempo real | **Não recomendado** para o caminho quente |
| **Java/Kotlin puro** | Mais simples, integração direta | Sem controle de memória fino; GC no caminho de inferência; é justamente o que se quer evitar | Só para a UI |

### 4.3 Ressalvas honestas sobre o ganho de performance

- **O ganho de C++ vem menos da linguagem e mais do caminho dos dados.** A maior parte do custo por quadro costuma ser cópia de imagem, alocação e troca entre threads. Isso se resolve com buffers pré-alocados, acesso direto ao frame da câmera e delegate do TFLite/LiteRT (GPU ou de fabricante; a NNAPI foi marcada como descontinuada a partir do Android 15, **verificar na documentação atual**). Escolher C++ sem isso não basta.
- **O núcleo C++ processa pouco dado se o MediaPipe ficar do lado Java.** Nesse cenário (seção 4.1), o C++ recebe 63 floats por quadro e roda duas CNNs pequenas; o caminho pesado (câmera → imagem → MediaPipe) fica fora dele. O ganho esperado do núcleo C++ é então **modesto**, e o argumento mais forte para ele passa a ser o reuso com o Pico e a paridade testável no host. Se o objetivo for ganho grande de latência, o ponto a otimizar é o caminho da câmera e do MediaPipe, não a linguagem do núcleo.
- **Não sabemos se o gargalo está na inferência.** Os modelos são CNNs pequenas (sobre 21 pontos e uma janela de 30 quadros). O custo pode estar no MediaPipe, na câmera ou na cópia. Medir em F2 antes de otimizar.
- **Controle de memória é um custo também.** Com C++ quem gerencia é o programador. Por isso o núcleo tem testes no host (Linux) com ASan/UBSan, antes de qualquer build Android.
- **Reuso com o Pico:** o núcleo de features e segmentação não depende de Android. Para o Raspberry Pi/Pico, a mesma lógica C++ serve, trocando apenas o backend de inferência (TFLite Micro no Pico). Isso é uma vantagem concreta de C++ sobre Kotlin neste projeto.
- **Kotlin ou Java na UI:** o código Java é aceito. Kotlin é a linguagem recomendada pelo Google para Android e compila para o mesmo bytecode; a diferença para este projeto é pequena. Escolher Java é válido se for a preferência do time.

### 4.4 Threading e ponte JNI

- A UI nunca espera inferência. A thread principal só recebe o último resultado pronto.
- O núcleo roda em uma thread de trabalho própria. Frames entram por um buffer de tamanho fixo; se o núcleo estiver ocupado, o quadro antigo é descartado (não se enfileira).
- A ponte JNI é fina: uma chamada por quadro, passando um ponteiro para o buffer, sem alocar objetos Java no caminho.

## 5. Decisões de arquitetura (com recomendação)

| Decisão | Opções | Recomendação | Status |
|---|---|---|---|
| Qual modelo estático no celular | (a) CNN int8 embarcada; (b) portar o Random Forest; (c) converter o LSTM | **(a)** | Recomendado |
| Detector de mão | (a) MediaPipe Hand Landmarker (Tasks); (b) detector próprio | **(a)**. O host usa a API legada `mp.solutions.hands` (mediapipe 0.10.14); o Android usa a API Tasks, com modelo e pipeline que podem diferir. **Semelhante, não idêntico**: a paridade é medida em F1. | Recomendado |
| Linguagem do núcleo de processamento | C++; C; Rust; Go | **C++** (seção 4.2) | **Recomendada; aguarda confirmação do usuário e validação por benchmark em F2** |
| Linguagem da UI | Kotlin; Java | Kotlin (ou Java, se preferido) | Em aberto |
| Framework de UI | Compose; Views | Compose | Recomendado |
| Onde mora a lógica de features e segmentação | Núcleo C++ | **Núcleo C++**, validado por fixtures | Depende da confirmação da linguagem |
| Fonte de verdade dos parâmetros | Hardcoded no núcleo; ou lido do `embedded_bundle.json` | **Ler do manifesto** sempre que possível | Recomendado |

## 6. Fases

### F0 — Pré-requisitos (bloqueante)
- Ter o dataset local em `dataset/static/` e `dataset/temporal/`.
- Ter TensorFlow instalado, com Python 3.11 ou 3.12 (`tensorflow==2.16.1` não tem wheel para Python 3.13).
- **Exportação TFLite int8: corrigida** (via SavedModel, `src/model_training/tflite_export.py`, com teste de regressão). Verificada com dados sintéticos em Python 3.11 e 3.12; falta rodá-la com o dataset real.
- Rodar `make embedded-train` e `make embedded-export`. Isso gera os `.tflite` e o `embedded_bundle.json`.
- Rodar `make embedded-check`. Se falhar, o bundle não serve. Parar aqui.
- Registrar as métricas dos dois modelos embarcados. Sem isso, não há o que testar no celular.

**Critério de saída:** `embedded-check` passa; métricas por classe registradas.

### F1 — Fixtures de paridade e núcleo C++ no host
- Script no host que gera um JSON com: sequências de landmarks de exemplo, features `wrist_relative` esperadas, probabilidades esperadas dos dois modelos e o token final.
- Implementar o núcleo C++ (features, `MotionDetector`, `TemporalPipeline`, regra híbrida) como biblioteca **independente de Android**, compilada e testada no host com CMake e CTest.
- Testes com ASan e UBSan no build de host.
- Tolerância numérica definida antes (ex.: diferença máxima de 1e-4 nas features, e igualdade do token final).

**Critério de saída:** testes do núcleo C++ passam no host contra os fixtures, sem sanitizer reclamando.

### F2 — Esqueleto do app e benchmark
- Projeto Android com CameraX, a ponte JNI, o núcleo C++ compilado via NDK, e overlay dos 21 pontos.
- Sem inferência ainda. Só mostra a mão detectada e o FPS.
- Escolher o artefato de runtime (TFLite ou o LiteRT, que é o nome atual) e **confirmar que ele executa os `.tflite` gerados pelo TensorFlow 2.16.1**. Não foi verificado.
- **Medir antes de otimizar**: tempo por quadro separado em captura, MediaPipe, features, inferência e cópias. Esse benchmark decide se a escolha de C++ se justifica para o gargalo real.

**Critério de saída:** roda em um celular real, com o benchmark por etapa registrado. A meta de FPS (inicialmente 15) é uma hipótese, a ser confirmada pela medição.

### F3 — Inferência estática no celular
- Carregar o `.tflite` estático no núcleo C++, aplicar a quantização e devolver a letra com confiança à UI.
- Respeitar o limiar de 0.75.

**Critério de saída:** para as letras do dataset, o app reproduz as probabilidades dos fixtures dentro da tolerância.

### F4 — Pipeline temporal (J e Z)
- Carregar o `.tflite` temporal e aplicar a regra híbrida da seção 3.4, dentro do núcleo C++.

**Critério de saída:** para vídeos gravados de J e Z, o app emite os mesmos tokens que o Python no mesmo vídeo.

### F5 — Teste de campo
- Protocolo com pelo menos 3 pessoas (a mesma exigência da FASE 1), variando mão, iluminação e distância.
- Registrar cada predição com timestamp e rótulo verdadeiro, em um CSV local para análise.
- Comparar com o host sobre as mesmas gravações.

**Critério de saída:** métricas por classe e taxa de rejeição registradas, separadas por pessoa.

## 7. Riscos

| Risco | Probabilidade | Mitigação |
|---|---|---|
| Exportação TFLite int8 quebrada | Era certa; **corrigida** | Falta validar com o dataset real em F0. |
| `z` do MediaPipe Android diferente do Python | Média (hipótese não verificada) | Fixtures de F1 detectam logo. |
| Perda de acurácia pela quantização int8 | Média | Comparar acurácia float vs int8 no `embedded-check`. |
| Bug de memória no núcleo C++ | Média | ASan/UBSan no host (F1); testes antes do build Android. |
| Build NDK/CMake mais trabalhoso que Kotlin puro | Alta | Núcleo testado no host antes; build do NDK isolado em F2. |
| Otimizar a linguagem e não o gargalo | Média | Benchmark por etapa em F2 antes de otimizar (seção 4.3). |
| Câmera frontal espelhada muda a mão dominante | Alta | Mirror augmentation já existe no treino; testar ambas as câmeras. |
| Modelo embarcado com acurácia baixa, sem métrica registrada | Desconhecida | F0 exige métricas antes de F3. |
| Dataset local ausente no seu ambiente | Desconhecida | Confirmar antes de começar. |
| Latência alta em celulares antigos | Média | Medir em pelo menos 2 aparelhos. |

## 8. Ambiente de build

- Container atual: Java, Gradle e Python 3.11/3.12 existem, mas **sem Android SDK nem NDK**. Não compila APK aqui.
- Verificado em 2026-11-01: este container tem `g++`, `clang++`, `cmake`, `make` e `ninja`. O núcleo C++ pode ser compilado e testado aqui, sem Android.
- Build Android: (a) Android Studio na sua máquina, com NDK; ou (b) CI no GitHub Actions com `setup-android` e NDK. A opção (a) é mais rápida para iterar; a (b) garante que o build não depende de uma máquina específica.

## 9. Critério de sucesso geral

- Paridade com o host dentro da tolerância definida em F1 e F4.
- Benchmark por etapa registrado em pelo menos 2 aparelhos, com decisão de otimização baseada nele.
- Métricas por classe registradas em teste de campo com pelo menos 3 pessoas.
- Build reproduzível por qualquer pessoa com o procedimento documentado.

## 10. Próximos passos

1. Confirmar a escolha de C++ para o núcleo e de Kotlin ou Java para a UI (seção 5).
2. Confirmar que o dataset local existe e onde está (F0).
3. Gerar o bundle com o dataset real (`make embedded-train`; a exportação int8 já foi corrigida) e rodar `embedded-check` (F0).
4. Só então começar F1.

Perguntas em aberto:
- O dataset local tem `dataset/static` e `dataset/temporal` populados?
- Você tem um Android físico para testes? Qual modelo e versão?
- Prefere Android Studio na sua máquina ou CI?
- UI em Kotlin ou Java?
