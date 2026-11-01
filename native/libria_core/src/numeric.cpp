#include "libria/numeric.h"

#include <cmath>

namespace libria {
namespace numeric {

// Mesmo algoritmo de @TYPE@_pairwise_sum do NumPy (loops_utils.h.src):
// laço simples abaixo de 8 elementos, 8 acumuladores até o bloco de 128 e
// divisão recursiva acima disso. A redução começa da identidade (0).
float pairwise_sum(const float* values, std::size_t count) noexcept {
    constexpr std::size_t kBlock = 128;
    if (count < 8) {
        float result = 0.0f;
        for (std::size_t i = 0; i < count; ++i) {
            result += values[i];
        }
        return result;
    }
    if (count <= kBlock) {
        float r[8];
        for (std::size_t j = 0; j < 8; ++j) {
            r[j] = values[j];
        }
        std::size_t i = 8;
        for (; i < count - (count % 8); i += 8) {
            for (std::size_t j = 0; j < 8; ++j) {
                r[j] += values[i + j];
            }
        }
        float result = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < count; ++i) {
            result += values[i];
        }
        return result;
    }
    std::size_t half = count / 2;
    half -= half % 8;
    return pairwise_sum(values, half) + pairwise_sum(values + half, count - half);
}

float mean(const float* values, std::size_t count) noexcept {
    return pairwise_sum(values, count) / static_cast<float>(count);
}

// std::nearbyint usa o modo de arredondamento corrente, que por padrão é
// "para o mais próximo, empate para o par" — o mesmo de np.rint.
float round_half_even(float value) noexcept { return std::nearbyint(value); }
double round_half_even(double value) noexcept { return std::nearbyint(value); }

}  // namespace numeric
}  // namespace libria
