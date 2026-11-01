#pragma once
// Arbitragem entre os modelos estático e temporal.
//
// Referência Python: src/inference/libras_embedded_runtime.py,
// choose_embedded_prediction().

#include <optional>
#include <string>
#include <vector>

namespace libria {

struct EmbeddedPrediction {
    std::string token;
    double confidence = 0.0;
    std::string source;
};

// O temporal vence se prevê uma classe prioritária (ex.: J, Z) com confiança
// >= temporal_threshold. Senão vale o estático com confiança >= static_threshold.
// Senão, nenhum. A comparação de rótulos ignora maiúsculas (ASCII).
std::optional<EmbeddedPrediction> choose_embedded_prediction(
    const std::optional<EmbeddedPrediction>& static_prediction,
    const std::optional<EmbeddedPrediction>& temporal_prediction, double static_threshold,
    double temporal_threshold, const std::vector<std::string>& temporal_priority_classes);

}  // namespace libria
