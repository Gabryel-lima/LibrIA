#include "libria/quantization.h"

#include <cmath>
#include <limits>

#include "libria/numeric.h"

namespace libria {

namespace {
constexpr float kInt8Min = static_cast<float>(std::numeric_limits<std::int8_t>::min());
constexpr float kInt8Max = static_cast<float>(std::numeric_limits<std::int8_t>::max());

std::int8_t saturate(float value) {
    if (std::isnan(value)) return 0;
    if (value < kInt8Min) return std::numeric_limits<std::int8_t>::min();
    if (value > kInt8Max) return std::numeric_limits<std::int8_t>::max();
    return static_cast<std::int8_t>(value);  // trunca em direção a zero, como astype
}
}  // namespace

void quantize_int8(const float* input, std::size_t count, QuantizationParams params,
                   std::int8_t* out) noexcept {
    if (params.scale == 0.0f) {
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = saturate(input[i]);
        }
        return;
    }
    // Como no NumPy: tensor float32 / escala (convertida para float32) + zero
    // point, arredondamento np.round (meio para o par) e clip.
    const float zero_point = static_cast<float>(params.zero_point);
    for (std::size_t i = 0; i < count; ++i) {
        const float scaled = input[i] / params.scale + zero_point;
        out[i] = saturate(numeric::round_half_even(scaled));
    }
}

float dequantize_normalize_int8(const std::int8_t* raw, std::size_t count,
                                QuantizationParams params, float* out) noexcept {
    const float zero_point = static_cast<float>(params.zero_point);
    for (std::size_t i = 0; i < count; ++i) {
        const float value = static_cast<float>(raw[i]);
        out[i] = params.scale == 0.0f ? value : (value - zero_point) * params.scale;
    }
    const float sum = numeric::pairwise_sum(out, count);
    if (sum > 0.0f) {
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = out[i] / sum;
        }
    }
    return sum;
}

}  // namespace libria
