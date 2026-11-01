#pragma once
// Detecção de início e fim de sinal, com histerese.
//
// Referência Python: src/inference/sign_segmenter.py, SignSegmenter.

#include <cstddef>
#include <optional>
#include <vector>

namespace libria {

enum class SegmentReason { kMotionStopped, kMaxDuration, kHandLost };

const char* to_string(SegmentReason reason) noexcept;

struct SignSegment {
    std::vector<float> frames;  // frame_count x feature_dim, contíguos
    std::size_t frame_count = 0;
    double start_time = 0.0;
    double end_time = 0.0;
    SegmentReason reason = SegmentReason::kMotionStopped;

    double duration_seconds() const noexcept {
        return end_time > start_time ? end_time - start_time : 0.0;
    }
};

struct SegmenterConfig {
    double start_threshold = 0.012;
    double end_threshold = 0.006;
    int min_start_frames = 2;
    int min_end_frames = 5;
    int min_frames = 4;
    double min_duration_seconds = 0.15;
    double max_duration_seconds = 4.0;
    int max_absent_frames = 5;
};

class SignSegmenter {
public:
    // Lança std::invalid_argument se end_threshold > start_threshold, se
    // min_start_frames/min_end_frames <= 0 ou se feature_dim == 0.
    SignSegmenter(const SegmenterConfig& config, std::size_t feature_dim);

    bool is_active() const noexcept { return active_; }
    std::size_t frame_count() const noexcept { return timestamps_.size(); }
    std::optional<double> start_time() const noexcept;

    void reset() noexcept;

    // `features == nullptr` ou `hand_present == false` significa mão ausente.
    // Devolve o segmento quando um sinal se encerra e é aceito.
    std::optional<SignSegment> update(const float* features, double timestamp, double energy,
                                      bool hand_present = true);

private:
    std::optional<SignSegment> update_idle(const float* features, double timestamp, double energy,
                                           bool hand_present);
    std::optional<SignSegment> update_active(const float* features, double timestamp, double energy,
                                             bool hand_present);
    std::optional<SignSegment> close(SegmentReason reason, int trim);

    SegmenterConfig config_;
    std::size_t feature_dim_;
    bool active_ = false;
    std::vector<float> frames_;
    std::vector<double> timestamps_;
    std::vector<float> pending_frames_;
    std::vector<double> pending_timestamps_;
    int quiet_streak_ = 0;
    int absent_streak_ = 0;
};

}  // namespace libria
