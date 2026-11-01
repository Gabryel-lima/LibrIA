#pragma once
// Quantização int8 de entrada e desquantização da saída dos modelos TFLite.
//
// Referência Python: src/inference/libras_embedded_runtime.py,
// QuantizedTFLiteModel._quantize_input, _dequantize_output e predict().

#include <cstddef>
#include <cstdint>

namespace libria {

struct QuantizationParams {
    float scale = 0.0f;           // o TFLite guarda a escala em float32
    std::int32_t zero_point = 0;
};

// q = clip(round_half_even(x / scale + zero_point), -128, 127).
// Com scale == 0, converte truncando (como numpy astype), com saturação para
// evitar comportamento indefinido fora do intervalo.
void quantize_int8(const float* input, std::size_t count, QuantizationParams params,
                   std::int8_t* out) noexcept;

// p = (q - zero_point) * scale (ou q, se scale == 0), depois normaliza para
// soma 1 se a soma for > 0. Devolve a soma antes da normalização.
float dequantize_normalize_int8(const std::int8_t* raw, std::size_t count,
                                QuantizationParams params, float* out) noexcept;

}  // namespace libria
