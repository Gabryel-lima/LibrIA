# Núcleo nativo (C++)

Base do app Android previsto em [`.github/plans/android_plan.md`](../.github/plans/android_plan.md) (fase F1): a lógica de reconhecimento do pipeline Python, portada para C++17, sem dependências externas e testada no host contra o próprio Python.

```text
native/
├── fixtures/parity_fixtures.json   gerado por scripts/export_parity_fixtures.py
├── libria_jni/                     ponte JNI para o Kotlin (ver android/README.md)
└── libria_core/                    biblioteca C++ (CMake)
    ├── include/libria/             API pública
    ├── src/
    └── tests/                      paridade (fixtures) + unidade
```

## O que está portado

| C++ (`libria::`) | Referência Python |
|---|---|
| `wrist_relative_features` | `utils/helpers.py`, `landmarks_to_relative` |
| `MotionDetector` | `src/inference/motion_detector.py` |
| `TemporalBuffer`, `resample_sequence` | `src/inference/temporal_buffer.py` |
| `SignSegmenter` | `src/inference/sign_segmenter.py` |
| `ProbabilitySmoother`, `DuplicateSuppressor` | `src/inference/prediction_smoother.py` |
| `TemporalPipeline`, `PipelineConfig` | `src/inference/temporal_pipeline.py`, `TEMPORAL_PIPELINE_CONFIG` |
| `SignToken` | `src/inference/sign_token.py` (sem tipo de sinal e tradução: ficam com quem consome o token) |
| `choose_embedded_prediction` | `src/inference/libras_embedded_runtime.py` |
| `quantize_int8`, `dequantize_normalize_int8` | `QuantizedTFLiteModel` (mesmo arquivo) |

**Fora do núcleo, por enquanto:** a inferência TFLite e o MediaPipe. Os modelos entram por injeção de dependência (`TemporalPredictor`, `StaticPredictor`), como no Python. Isso mantém o núcleo testável no host e permite trocar LSTM, CNN temporal ou Transformer sem tocar no pipeline.

## Como compilar e testar

```bash
make native-test        # ASan + UBSan + -Werror, roda a paridade
make native-fixtures    # regenera as fixtures a partir do pipeline Python
```

Sem o Makefile:

```bash
cmake -S native/libria_core -B native/libria_core/build -DLIBRIA_SANITIZE=ON -DLIBRIA_WERROR=ON
cmake --build native/libria_core/build
cd native/libria_core/build && ctest --output-on-failure
```

A suíte Python também compila e roda o núcleo (`tests/test_native_core.py`, pulado sem CMake) e confere que as fixtures estão em dia com o pipeline (`tests/test_parity_fixtures.py`).

## Paridade com o Python: como funciona

1. `scripts/export_parity_fixtures.py` roda o **código Python real** sobre entradas determinísticas e grava entradas e saídas em JSON. Nos cenários do pipeline, cada chamada aos preditores fica gravada com os índices dos quadros recebidos e a resposta devolvida.
2. Os testes C++ leem o JSON e exigem **igualdade exata, bit a bit**: mesmas features, mesmas energias, mesmos tokens, e os preditores consultados nos mesmos quadros com as mesmas janelas.
3. Se o pipeline Python mudar, `tests/test_parity_fixtures.py` falha até as fixtures serem regeneradas; aí `make native-test` mostra se o C++ precisa acompanhar.
4. O teste nativo também falha se aparecer uma chave nova em `TEMPORAL_PIPELINE_CONFIG`, ou se os valores padrão de `PipelineConfig` divergirem de `config/settings.py`.

### Por que bit a bit, e o que isso exige

Uma energia de movimento perto de um limiar decide se um sinal começou. Se o C++ arredondasse diferente do NumPy, o celular e o Python discordariam justamente nos casos de fronteira. Para a igualdade exata, o núcleo reproduz o NumPy:

- aritmética em float32 onde o NumPy usa float32 (features, energia bruta, quantização) e em double onde usa float64 (média móvel, suavização);
- soma *pairwise* do NumPy em `np.mean`/`np.sum` (`numeric::pairwise_sum`), começando da identidade 0, como confirmado empiricamente no NumPy 1.26 e 2.x;
- arredondamento de `np.rint`/`np.round` (meio para o par) via `std::nearbyint`;
- **`-ffp-contract=off`**: o compilador não pode fundir multiplicação e soma em FMA. Medido: compilando com FMA ligado (`-mfma -ffp-contract=fast`), 3 dos 14 testes falham, inclusive os cenários do pipeline. No ARM64 do Android a fusão acontece por padrão, então a flag é obrigatória.

A suíte foi validada por mutação: alterar cada um desses detalhes (soma, arredondamento, ordem da norma, cooldown, renovação do supressor, recorte do segmento, normalização de rótulo) faz algum teste falhar. As duas únicas mutações não detectadas são equivalentes (não mudam o resultado).

## Uso

```cpp
#include "libria/features.h"
#include "libria/temporal_pipeline.h"

libria::TemporalPipeline pipeline(
    /* temporal */ [&](const float* seq, std::size_t frames, std::size_t dim, std::vector<double>& probs) {
        run_temporal_model(seq, frames, dim, probs);   // ex.: TFLite int8
    },
    /* label_map */ { {0, "J"}, {1, "Z"} },  // o espaço entre as chaves evita o Liquid do GitHub Pages
    /* sequence_length */ 30,
    /* feature_dim */ libria::kFeatureDimension,
    /* static */ [&](const float* features, std::size_t dim) {
        return libria::StaticPrediction{run_static_model(features, dim)};
    });

float features[libria::kFeatureDimension];
libria::wrist_relative_features(landmarks /* 21 x (x, y, z) */, features);
if (auto token = pipeline.process_frame(hand_detected ? features : nullptr, timestamp_seconds)) {
    // token->label, token->confidence, token->state (partial/final/rejected)
}
```

## Premissas e limites

- **Rótulos ASCII.** A normalização (`strip` + maiúsculas) trata só ASCII. Os rótulos de `config/vocabulary.py` são ASCII; os acentos aparecem só nas traduções.
- **Entradas finitas.** Com NaN nas features, o resultado não acompanha o Python.
- **Não é thread-safe.** Use um `TemporalPipeline` por thread.
- **Sem alocação por quadro é objetivo de projeto, não medição.** Os buffers são reutilizados; há alocação ao fechar um segmento e em rótulos com mais de 15 caracteres (fora do SSO do `std::string`).
- **Testado no host:** GCC 13 com ASan/UBSan e `-Werror`; Clang 18 com `-Werror` (os runtimes de sanitizer do Clang não estão instalados neste ambiente). Ainda **não** compilado com o NDK nem executado em Android.

## Próximos passos (plano Android)

- F0: gerar os `.tflite` com o dataset real (a exportação int8 já foi corrigida).
- F2 (em andamento): a ponte JNI (`native/libria_jni`) e a API Kotlin já existem e passam na JVM; o app Android está escrito, falta compilar e medir no aparelho. Ver [`android/README.md`](../android/README.md).
- F3/F4: ligar os preditores à inferência TFLite (usando `quantize_int8`/`dequantize_normalize_int8`) e validar contra os fixtures no aparelho.
