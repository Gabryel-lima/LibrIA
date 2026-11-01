#pragma once
// Primitivas numéricas que reproduzem o NumPy bit a bit.
//
// O pipeline Python é a referência de comportamento. Para que uma energia perto
// de um limiar dê a mesma decisão aqui e lá, as somas seguem o mesmo algoritmo
// do NumPy (soma em pares, "pairwise") e o arredondamento segue np.rint
// (meio para o par). A biblioteca é compilada com -ffp-contract=off para o
// compilador não fundir multiplicação e soma (FMA), o que mudaria os bits.

#include <cstddef>

namespace libria {
namespace numeric {

// Soma em float32 com o algoritmo pairwise do NumPy (np.add.reduce em float32).
float pairwise_sum(const float* values, std::size_t count) noexcept;

// np.mean de um vetor float32: soma pairwise, depois divisão em float32.
float mean(const float* values, std::size_t count) noexcept;

// np.rint: arredonda para o inteiro mais próximo, empates para o par.
float round_half_even(float value) noexcept;
double round_half_even(double value) noexcept;

}  // namespace numeric
}  // namespace libria
