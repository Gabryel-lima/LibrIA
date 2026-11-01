# App Android (fase F2)

Projeto Gradle do app que testa o reconhecimento de Libras no celular. Plano completo em [`.github/plans/android_plan.md`](../.github/plans/android_plan.md).

```text
android/
├── libria-bridge/   Kotlin/JVM: API do núcleo C++ via JNI, benchmark por etapa, FPS, overlay
└── app/             App Android: CameraX + MediaPipe Hand Landmarker + Compose
native/
├── libria_core/     núcleo C++ (F1)
└── libria_jni/      ponte JNI, compilada para o Android (NDK) e para o host (JDK)
```

## O que está verificado e o que não está

| Parte | Estado |
|---|---|
| Ponte JNI (`native/libria_jni`) | **Verificada** no host: compila com `-Werror` e roda na JVM com `-Xcheck:jni`, sem avisos. |
| `libria-bridge` (Kotlin) | **Verificado**: 14 testes. Os 14 cenários de paridade com o Python passam **pelo JNI**, bit a bit, com os preditores em Kotlin chamados de volta pelo C++. Inclui propagação de exceções, erros de argumento, uso após `close` e 500 ciclos de criar/fechar sem vazar referências. Validado por mutação (ordem dos campos de configuração e do enum de estado). |
| `app` (Android) | **Escrito, nunca compilado.** O ambiente em que foi criado bloqueia `dl.google.com`, de onde vêm o Android SDK, o NDK e todo o Google Maven (AndroidX, CameraX, Compose, MediaPipe Tasks). Uma checagem de tipos parcial, sem as bibliotecas Android, confirmou só que o app usa a API do `libria-bridge` corretamente. |
| Versões das dependências do app | **Não verificadas.** Escolhidas de memória, de releases que existiam até o fim de 2024 (AGP 8.7.3, Kotlin 2.0.21, Compose BOM 2024.12.01, CameraX 1.4.1, MediaPipe Tasks 0.10.14, NDK 27.2.12479018). São intencionalmente conservadoras para maximizar a chance de existirem; atualize depois de compilar. |

## Testar o que roda sem Android

```bash
make native-bridge-test          # ou: cd android && ./gradlew :libria-bridge:test
```

Precisa de JDK 17+, CMake e um compilador C++.

> No ambiente cloud em que este módulo foi criado, o Maven Central devolvia 429 (limite de requisições) de forma intermitente. O contorno foi um *init script* local, fora do repositório, que consulta antes o espelho do Google Cloud Storage:
>
> ```kotlin
> // mirror.init.gradle.kts — uso: ./gradlew --init-script mirror.init.gradle.kts :libria-bridge:test
> val mirror = "https://maven-central.storage-download.googleapis.com/maven2/"
> beforeSettings {
>     pluginManagement.repositories.maven { url = uri(mirror) }
>     dependencyResolutionManagement.repositories.maven { url = uri(mirror) }
> }
> ``` O Gradle compila `native/libria_jni` para o host e roda os testes na JVM. Sem Android SDK, o módulo `:app` fica fora do build automaticamente (o `settings.gradle.kts` só o inclui com `ANDROID_HOME` ou `sdk.dir` em `local.properties`).

## Compilar o app

Requisitos: Android SDK com a plataforma 35, NDK `27.2.12479018` e CMake `3.22.1` do SDK.

```bash
cd android
echo "sdk.dir=$HOME/Android/Sdk" > local.properties   # ou exporte ANDROID_HOME
./gradlew :app:assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

O primeiro build baixa o modelo `hand_landmarker.task` do MediaPipe (~7,8 MB) para `app/src/main/assets/` (não versionado).

**No ambiente cloud do Claude**, o build do app só funciona com `dl.google.com` liberado na política de rede do ambiente (o `maven.google.com` redireciona para ele).

## O que o app faz na F2

- Câmera frontal (CameraX) → MediaPipe Hand Landmarker (`LIVE_STREAM`, 1 mão, mesmos limiares do Python) → features `wrist_relative` e pipeline temporal, ambos no núcleo C++.
- **Ainda sem modelo de Libras**: o pipeline roda com um preditor uniforme. Isso já exercita, no aparelho real, a detecção de movimento e a segmentação de sinais (sinais encerrados saem como `DESCONHECIDO`).
- Mostra os 21 pontos sobre a imagem, FPS, energia de movimento, se há sinal em execução, e o **tempo por etapa** (p50/p95): `conversao` (ImageProxy → Bitmap), `mediapipe`, `features` e `nucleo`. Essa medição é o objetivo da F2: decidir onde otimizar com dados, não com suposição.

## Decisões e premissas

- **A imagem enviada ao MediaPipe não é espelhada**, porque a coleta em Python (`scripts/collect_dataset.py`) também não espelha o quadro. Só o desenho na tela é espelhado, como a pré-visualização da câmera frontal. **Hipótese a confirmar no aparelho**: depende de como a webcam usada na coleta entregava a imagem.
- **MediaPipe Tasks 0.10.14**, a mesma versão do MediaPipe Python fixada no projeto. O host usa a API legada (`mp.solutions.hands`) e o app usa a API Tasks; os landmarks podem diferir mesmo com a versão igual. A paridade das features com dados reais é um item da F3.
- **Preditores em Kotlin, chamados pelo C++.** Deixa a inferência TFLite trocável (Kotlin/LiteRT ou C++) sem mexer na ponte. O custo é uma chamada JNI por consulta ao modelo: a cada 5 quadros durante um sinal, e a cada 20 quadros parados.
- **Bibliotecas nativas com páginas de 16 KB** (`-Wl,-z,max-page-size=16384`), exigência para aparelhos com Android 15+.

## Próximos passos

1. Liberar `dl.google.com` (ou compilar fora do ambiente cloud) e fazer o app compilar; corrigir o que aparecer.
2. Rodar em pelo menos 2 aparelhos e registrar o tempo por etapa (critério de saída da F2).
3. Confirmar a hipótese do espelhamento.
4. F3: ligar os preditores ao TFLite com os `.tflite` da F0.
