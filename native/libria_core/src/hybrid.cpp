#include "libria/hybrid.h"

#include <algorithm>

namespace libria {

namespace {
std::string upper_ascii(std::string value) {
    for (char& c : value) {
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    }
    return value;
}
}  // namespace

std::optional<EmbeddedPrediction> choose_embedded_prediction(
    const std::optional<EmbeddedPrediction>& static_prediction,
    const std::optional<EmbeddedPrediction>& temporal_prediction, double static_threshold,
    double temporal_threshold, const std::vector<std::string>& temporal_priority_classes) {
    if (temporal_prediction.has_value()) {
        const std::string token = upper_ascii(temporal_prediction->token);
        const bool priority = std::any_of(
            temporal_priority_classes.begin(), temporal_priority_classes.end(),
            [&token](const std::string& label) { return upper_ascii(label) == token; });
        if (priority && temporal_prediction->confidence >= temporal_threshold) {
            return temporal_prediction;
        }
    }
    if (static_prediction.has_value() && static_prediction->confidence >= static_threshold) {
        return static_prediction;
    }
    return std::nullopt;
}

}  // namespace libria
