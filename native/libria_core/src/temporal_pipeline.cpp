#include "libria/temporal_pipeline.h"

#include <stdexcept>
#include <utility>

namespace libria {

namespace {

SegmenterConfig segmenter_config(const PipelineConfig& config) {
    SegmenterConfig result;
    result.start_threshold = config.motion_start_threshold;
    result.end_threshold = config.motion_end_threshold;
    result.min_start_frames = config.min_start_frames;
    result.min_end_frames = config.min_end_frames;
    result.min_frames = config.min_segment_frames;
    result.min_duration_seconds = config.min_duration_seconds;
    result.max_duration_seconds = config.max_duration_seconds;
    result.max_absent_frames = config.max_absent_frames;
    return result;
}

std::size_t positive(int value, const char* name) {
    if (value <= 0) {
        throw std::invalid_argument(std::string(name) + " deve ser maior que zero");
    }
    return static_cast<std::size_t>(value);
}

SignToken make_token(const std::string& label, double confidence, double start_time,
                     double end_time, TokenSource source, TokenState state, int frame_count) {
    SignToken token;
    token.label = normalize_label(label);
    token.confidence = confidence;
    token.start_time = start_time;
    token.end_time = end_time;
    token.source = source;
    token.state = state;
    token.frame_count = frame_count;
    return token;
}

}  // namespace

TemporalPipeline::TemporalPipeline(TemporalPredictor temporal_predictor,
                                   std::map<int, std::string> label_map,
                                   std::size_t sequence_length, std::size_t feature_dim,
                                   StaticPredictor static_predictor, PipelineConfig config)
    : config_(std::move(config)),
      temporal_predictor_(std::move(temporal_predictor)),
      static_predictor_(std::move(static_predictor)),
      label_map_(std::move(label_map)),
      sequence_length_(sequence_length),
      feature_dim_(feature_dim),
      buffer_(sequence_length, feature_dim),
      motion_(config_.motion_smoothing),
      segmenter_(segmenter_config(config_), feature_dim),
      smoother_(positive(config_.smoothing_window, "smoothing_window")),
      suppressor_(config_.duplicate_window_seconds) {
    if (!temporal_predictor_) {
        throw std::invalid_argument("temporal_predictor é obrigatório");
    }
    positive(config_.partial_interval_frames, "partial_interval_frames");
    positive(config_.static_interval_frames, "static_interval_frames");
    sequence_scratch_.resize(sequence_length_ * feature_dim_);
}

void TemporalPipeline::reset() {
    buffer_.reset();
    motion_.reset();
    segmenter_.reset();
    smoother_.reset();
    suppressor_.reset();
    frame_index_ = 0;
    last_energy_ = 0.0;
    last_token_.reset();
    static_blocked_until_ = 0.0;
}

std::optional<SignToken> TemporalPipeline::process_frame(const float* features, double timestamp,
                                                         bool hand_present) {
    const bool present = hand_present && features != nullptr;
    const float* frame = present ? features : nullptr;
    ++frame_index_;
    last_energy_ = motion_.update(frame, feature_dim_);

    if (present) {
        buffer_.append(features, timestamp);
    }

    std::optional<SignSegment> segment = segmenter_.update(frame, timestamp, last_energy_, present);
    if (segment.has_value()) {
        return emit(finalize_segment(*segment));
    }
    if (segmenter_.is_active()) {
        return emit(partial_token(timestamp));
    }
    return emit(static_token(frame, timestamp));
}

std::optional<SignToken> TemporalPipeline::emit(std::optional<SignToken> token) {
    if (token.has_value()) {
        last_token_ = token;
    }
    return token;
}

void TemporalPipeline::predict_sequence(const float* sequence) {
    probabilities_.clear();
    temporal_predictor_(sequence, sequence_length_, feature_dim_, probabilities_);
    if (probabilities_.empty()) {
        throw std::runtime_error("temporal_predictor devolveu uma distribuição vazia");
    }
}

std::string TemporalPipeline::label_for(std::size_t index) const {
    // Python: str(label_map.get(index, index)).upper() — maiúsculas, sem strip.
    // O strip só acontece ao montar o token; o supressor recebe este valor.
    const auto found = label_map_.find(static_cast<int>(index));
    std::string label = found != label_map_.end() ? found->second : std::to_string(index);
    for (char& c : label) {
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    }
    return label;
}

std::optional<SignToken> TemporalPipeline::partial_token(double timestamp) {
    // Hipótese durante a execução do sinal, para feedback imediato.
    if (!config_.emit_partial_tokens) {
        return std::nullopt;
    }
    if (frame_index_ % static_cast<std::uint64_t>(config_.partial_interval_frames) != 0) {
        return std::nullopt;
    }
    if (!buffer_.window(sequence_length_, sequence_scratch_.data())) {
        return std::nullopt;
    }

    predict_sequence(sequence_scratch_.data());
    smoother_.update(probabilities_);
    const auto best = smoother_.best();
    if (!best.has_value()) {
        return std::nullopt;
    }

    const std::optional<double> start = segmenter_.start_time();
    return make_token(label_for(best->first), best->second, start.value_or(timestamp), timestamp,
                      TokenSource::kTemporal, TokenState::kPartial,
                      static_cast<int>(segmenter_.frame_count()));
}

std::optional<SignToken> TemporalPipeline::finalize_segment(const SignSegment& segment) {
    // Classifica o segmento fechado e decide entre aceitar e recusar.
    resample_sequence(segment.frames.data(), segment.frame_count, feature_dim_, sequence_length_,
                      index_scratch_, sequence_scratch_.data());
    // A predição do segmento completo entra na mesma janela das parciais: a
    // decisão final considera todo o sinal, não só o último instante.
    predict_sequence(sequence_scratch_.data());
    smoother_.update(probabilities_);
    const auto best = smoother_.best();
    smoother_.reset();
    if (!best.has_value()) {
        return std::nullopt;
    }

    const std::string label = label_for(best->first);
    const double confidence = best->second;
    const bool rejected = confidence < config_.temporal_confidence_threshold;
    const std::string& emitted_label = rejected ? config_.unknown_label : label;

    // O retorno da mão ao repouso não pode ser lido como letra.
    static_blocked_until_ = segment.end_time + config_.static_cooldown_seconds;

    if (!suppressor_.accept(emitted_label, segment.end_time)) {
        return std::nullopt;
    }

    return make_token(emitted_label, confidence, segment.start_time, segment.end_time,
                      TokenSource::kTemporal, rejected ? TokenState::kRejected : TokenState::kFinal,
                      static_cast<int>(segment.frame_count));
}

std::optional<SignToken> TemporalPipeline::static_token(const float* features, double timestamp) {
    // Fallback estático: a mão parada ainda pode estar formando uma letra.
    if (!static_predictor_ || features == nullptr) {
        return std::nullopt;
    }
    // Letra é mão parada: sem leitura de movimento confiável, ou com a mão
    // ainda em deslocamento, o modelo estático não deve opinar.
    if (!motion_.has_reading()) {
        return std::nullopt;
    }
    if (last_energy_ >= config_.motion_end_threshold) {
        return std::nullopt;
    }
    if (timestamp < static_blocked_until_) {
        return std::nullopt;
    }
    if (frame_index_ % static_cast<std::uint64_t>(config_.static_interval_frames) != 0) {
        return std::nullopt;
    }

    const StaticPrediction prediction = static_predictor_(features, feature_dim_);
    if (prediction.confidence < config_.static_confidence_threshold) {
        return std::nullopt;
    }

    const std::string normalized = normalize_label(prediction.label);
    if (!suppressor_.accept(normalized, timestamp)) {
        return std::nullopt;
    }

    return make_token(normalized, prediction.confidence, timestamp, timestamp, TokenSource::kStatic,
                      TokenState::kFinal, 1);
}

}  // namespace libria
