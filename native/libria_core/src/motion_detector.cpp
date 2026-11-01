#include "libria/motion_detector.h"

#include <cmath>
#include <stdexcept>

#include "libria/numeric.h"

namespace libria {

MotionDetector::MotionDetector(double smoothing, std::size_t coordinates_per_point)
    : smoothing_(smoothing), coordinates_per_point_(coordinates_per_point) {
    if (!(smoothing >= 0.0 && smoothing < 1.0)) {
        throw std::invalid_argument("smoothing deve estar em [0.0, 1.0)");
    }
    if (coordinates_per_point == 0) {
        throw std::invalid_argument("coordinates_per_point deve ser maior que zero");
    }
}

void MotionDetector::reset() noexcept {
    has_previous_ = false;
    energy_ = 0.0;
    consecutive_frames_ = 0;
}

double MotionDetector::raw_energy(const float* features, std::size_t size) {
    if (!has_previous_ || previous_.size() != size) {
        return 0.0;
    }
    if (size % coordinates_per_point_ == 0) {
        // Deslocamento euclidiano médio por ponto.
        const std::size_t points = size / coordinates_per_point_;
        scratch_.resize(points);
        for (std::size_t p = 0; p < points; ++p) {
            float squared = 0.0f;
            for (std::size_t c = 0; c < coordinates_per_point_; ++c) {
                const std::size_t k = p * coordinates_per_point_ + c;
                const float delta = features[k] - previous_[k];
                squared += delta * delta;
            }
            scratch_[p] = std::sqrt(squared);
        }
        return static_cast<double>(numeric::mean(scratch_.data(), points));
    }
    scratch_.resize(size);
    for (std::size_t k = 0; k < size; ++k) {
        scratch_[k] = std::fabs(features[k] - previous_[k]);
    }
    return static_cast<double>(numeric::mean(scratch_.data(), size));
}

double MotionDetector::update(const float* features, std::size_t size) {
    if (features == nullptr) {
        has_previous_ = false;
        consecutive_frames_ = 0;
        energy_ *= smoothing_;
        return energy_;
    }

    const double raw = raw_energy(features, size);
    previous_.assign(features, features + size);
    has_previous_ = true;
    ++consecutive_frames_;

    energy_ = smoothing_ * energy_ + (1.0 - smoothing_) * raw;
    return energy_;
}

}  // namespace libria
