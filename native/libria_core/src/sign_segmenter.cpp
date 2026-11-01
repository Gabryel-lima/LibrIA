#include "libria/sign_segmenter.h"

#include <stdexcept>
#include <utility>

namespace libria {

const char* to_string(SegmentReason reason) noexcept {
    switch (reason) {
        case SegmentReason::kMotionStopped: return "motion_stopped";
        case SegmentReason::kMaxDuration: return "max_duration";
        case SegmentReason::kHandLost: return "hand_lost";
    }
    return "unknown";
}

SignSegmenter::SignSegmenter(const SegmenterConfig& config, std::size_t feature_dim)
    : config_(config), feature_dim_(feature_dim) {
    if (config.end_threshold > config.start_threshold) {
        throw std::invalid_argument(
            "end_threshold deve ser menor ou igual a start_threshold (histerese)");
    }
    if (config.min_start_frames <= 0 || config.min_end_frames <= 0) {
        throw std::invalid_argument("min_start_frames e min_end_frames devem ser maiores que zero");
    }
    if (feature_dim == 0) {
        throw std::invalid_argument("feature_dim deve ser maior que zero");
    }
}

std::optional<double> SignSegmenter::start_time() const noexcept {
    if (timestamps_.empty()) {
        return std::nullopt;
    }
    return timestamps_.front();
}

void SignSegmenter::reset() noexcept {
    active_ = false;
    frames_.clear();
    timestamps_.clear();
    pending_frames_.clear();
    pending_timestamps_.clear();
    quiet_streak_ = 0;
    absent_streak_ = 0;
}

std::optional<SignSegment> SignSegmenter::update(const float* features, double timestamp,
                                                 double energy, bool hand_present) {
    if (!active_) {
        return update_idle(features, timestamp, energy, hand_present);
    }
    return update_active(features, timestamp, energy, hand_present);
}

std::optional<SignSegment> SignSegmenter::update_idle(const float* features, double timestamp,
                                                      double energy, bool hand_present) {
    if (!hand_present || features == nullptr || energy < config_.start_threshold) {
        pending_frames_.clear();
        pending_timestamps_.clear();
        return std::nullopt;
    }

    // Acumula desde o primeiro quadro acima do limiar: o começo do sinal não
    // pode ser descartado enquanto confirmamos que é mesmo um sinal.
    pending_frames_.insert(pending_frames_.end(), features, features + feature_dim_);
    pending_timestamps_.push_back(timestamp);

    if (pending_timestamps_.size() >= static_cast<std::size_t>(config_.min_start_frames)) {
        active_ = true;
        frames_.swap(pending_frames_);
        timestamps_.swap(pending_timestamps_);
        pending_frames_.clear();
        pending_timestamps_.clear();
        quiet_streak_ = 0;
        absent_streak_ = 0;
    }
    return std::nullopt;
}

std::optional<SignSegment> SignSegmenter::update_active(const float* features, double timestamp,
                                                        double energy, bool hand_present) {
    if (hand_present && features != nullptr) {
        frames_.insert(frames_.end(), features, features + feature_dim_);
        timestamps_.push_back(timestamp);
        absent_streak_ = 0;
    } else {
        ++absent_streak_;
        if (absent_streak_ >= config_.max_absent_frames) {
            return close(SegmentReason::kHandLost, 0);
        }
    }

    if (energy < config_.end_threshold) {
        ++quiet_streak_;
        if (quiet_streak_ >= config_.min_end_frames) {
            // Os quadros parados do fim não pertencem ao sinal.
            return close(SegmentReason::kMotionStopped, quiet_streak_);
        }
    } else {
        quiet_streak_ = 0;
    }

    if (!timestamps_.empty() && (timestamp - timestamps_.front()) >= config_.max_duration_seconds) {
        return close(SegmentReason::kMaxDuration, 0);
    }
    return std::nullopt;
}

std::optional<SignSegment> SignSegmenter::close(SegmentReason reason, int trim) {
    SignSegment segment;
    segment.frames = std::move(frames_);
    std::vector<double> timestamps = std::move(timestamps_);
    std::size_t count = timestamps.size();

    const std::size_t min_frames = config_.min_frames > 0 ? static_cast<std::size_t>(config_.min_frames) : 0;
    if (trim > 0) {
        const std::size_t t = static_cast<std::size_t>(trim);
        // Python: `len(frames) - trim >= min_frames` (len - trim pode ser negativo).
        if (count >= t && count - t >= min_frames) {
            count -= t;
        }
    }

    reset();
    // reset() limpa frames_/timestamps_; o espaço movido para o segmento será
    // realocado no próximo sinal, o que é aceitável (uma vez por sinal).

    if (count < min_frames || count == 0) {
        return std::nullopt;
    }
    const double duration = timestamps[count - 1] - timestamps[0];
    if (duration < config_.min_duration_seconds) {
        return std::nullopt;
    }

    segment.frames.resize(count * feature_dim_);
    segment.frame_count = count;
    segment.start_time = timestamps[0];
    segment.end_time = timestamps[count - 1];
    segment.reason = reason;
    return segment;
}

}  // namespace libria
