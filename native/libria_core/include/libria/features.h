#pragma once
// Landmarks da mão -> vetor de features `wrist_relative`.
//
// Referência Python: utils/helpers.py, landmarks_to_relative().

#include <cstddef>

namespace libria {

constexpr std::size_t kLandmarkCount = 21;
constexpr std::size_t kCoordinatesPerLandmark = 3;
constexpr std::size_t kFeatureDimension = kLandmarkCount * kCoordinatesPerLandmark;  // 63

// Converte 21 landmarks (x, y, z intercalados, 63 floats) em features relativas
// ao pulso (landmark 0), divididas pela maior distância ao pulso.
// Se todos os pontos coincidem com o pulso, devolve zeros (sem divisão).
//
// `landmarks` e `out` têm kFeatureDimension floats e podem ser o mesmo buffer.
// As entradas devem ser finitas: com NaN o resultado não acompanha o Python.
void wrist_relative_features(const float* landmarks, float* out) noexcept;

}  // namespace libria
