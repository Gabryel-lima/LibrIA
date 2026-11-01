#include "libria/features.h"

#include <cmath>

namespace libria {

void wrist_relative_features(const float* landmarks, float* out) noexcept {
    const float wrist_x = landmarks[0];
    const float wrist_y = landmarks[1];
    const float wrist_z = landmarks[2];

    // Como no Python: diferenças e normas em float32. A norma é
    // sqrt((x*x + y*y) + z*z), a ordem de np.linalg.norm(axis=1).
    float max_distance = 0.0f;
    for (std::size_t i = 0; i < kLandmarkCount; ++i) {
        const float x = landmarks[3 * i] - wrist_x;
        const float y = landmarks[3 * i + 1] - wrist_y;
        const float z = landmarks[3 * i + 2] - wrist_z;
        out[3 * i] = x;
        out[3 * i + 1] = y;
        out[3 * i + 2] = z;
        const float distance = std::sqrt((x * x + y * y) + z * z);
        if (distance > max_distance) {
            max_distance = distance;
        }
    }

    if (max_distance > 0.0f) {
        for (std::size_t k = 0; k < kFeatureDimension; ++k) {
            out[k] = out[k] / max_distance;
        }
    }
}

}  // namespace libria
