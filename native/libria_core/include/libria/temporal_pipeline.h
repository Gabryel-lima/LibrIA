#pragma once
// Pipeline temporal: features -> buffer -> movimento -> segmentação -> modelo
// -> suavização -> supressão de duplicatas -> SignToken.
//
// Referência Python: src/inference/temporal_pipeline.py, TemporalPipeline.
//
// Os modelos entram por injeção de dependência (preditores), como no Python.
// Isso mantém o núcleo independente de TFLite e testável no host.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "libria/motion_detector.h"
#include "libria/prediction_smoother.h"
#include "libria/sign_segmenter.h"
#include "libria/sign_token.h"
#include "libria/temporal_buffer.h"

namespace libria {

// Espelha TEMPORAL_PIPELINE_CONFIG (config/settings.py). Os valores padrão são
// os do Python; o teste de paridade confere que continuam iguais.
struct PipelineConfig {
    double motion_smoothing = 0.6;
    double motion_start_threshold = 0.012;
    double motion_end_threshold = 0.006;
    int min_start_frames = 2;
    int min_end_frames = 5;
    int min_segment_frames = 8;
    double min_duration_seconds = 0.2;
    double max_duration_seconds = 4.0;
    int max_absent_frames = 5;
    int smoothing_window = 5;
    double duplicate_window_seconds = 1.0;
    bool emit_partial_tokens = true;
    int partial_interval_frames = 5;
    double temporal_confidence_threshold = 0.85;
    double static_confidence_threshold = 0.75;
    int static_interval_frames = 20;
    double static_cooldown_seconds = 0.5;
    // config/vocabulary.py, UNKNOWN_LABEL.
    std::string unknown_label = "DESCONHECIDO";
};

// Preditor temporal: recebe `frames x feature_dim` floats contíguos e escreve a
// distribuição de probabilidade em `probabilities` (o vetor é reutilizado entre
// chamadas para evitar alocação).
using TemporalPredictor = std::function<void(const float* sequence, std::size_t frames,
                                             std::size_t feature_dim,
                                             std::vector<double>& probabilities)>;

struct StaticPrediction {
    std::string label;
    double confidence = 0.0;
};

// Preditor estático: recebe as features de um quadro.
using StaticPredictor = std::function<StaticPrediction(const float* features, std::size_t feature_dim)>;

class TemporalPipeline {
public:
    // `label_map`: índice da classe -> rótulo. Índices ausentes viram o próprio
    // número. `static_predictor` pode ser vazio (sem fallback estático).
    // Lança std::invalid_argument para configuração inválida.
    TemporalPipeline(TemporalPredictor temporal_predictor, std::map<int, std::string> label_map,
                     std::size_t sequence_length, std::size_t feature_dim,
                     StaticPredictor static_predictor = nullptr, PipelineConfig config = {});

    // Processa um quadro. `features == nullptr` (ou hand_present == false)
    // significa mão ausente. Devolve um token quando houver o que reportar.
    // Lança std::runtime_error se o preditor temporal devolver distribuição vazia.
    std::optional<SignToken> process_frame(const float* features, double timestamp,
                                           bool hand_present = true);

    void reset();

    bool is_signing() const noexcept { return segmenter_.is_active(); }
    double last_energy() const noexcept { return last_energy_; }
    const std::optional<SignToken>& last_token() const noexcept { return last_token_; }
    const PipelineConfig& config() const noexcept { return config_; }
    std::size_t feature_dim() const noexcept { return feature_dim_; }
    std::size_t sequence_length() const noexcept { return sequence_length_; }

private:
    std::optional<SignToken> emit(std::optional<SignToken> token);
    void predict_sequence(const float* sequence);
    std::string label_for(std::size_t index) const;
    std::optional<SignToken> partial_token(double timestamp);
    std::optional<SignToken> finalize_segment(const SignSegment& segment);
    std::optional<SignToken> static_token(const float* features, double timestamp);

    PipelineConfig config_;
    TemporalPredictor temporal_predictor_;
    StaticPredictor static_predictor_;
    std::map<int, std::string> label_map_;
    std::size_t sequence_length_;
    std::size_t feature_dim_;

    TemporalBuffer buffer_;
    MotionDetector motion_;
    SignSegmenter segmenter_;
    ProbabilitySmoother smoother_;
    DuplicateSuppressor suppressor_;

    std::uint64_t frame_index_ = 0;
    double last_energy_ = 0.0;
    std::optional<SignToken> last_token_;
    double static_blocked_until_ = 0.0;

    // Buffers reutilizados: nenhuma alocação por quadro em regime estável.
    std::vector<float> sequence_scratch_;
    std::vector<std::size_t> index_scratch_;
    std::vector<double> probabilities_;
};

}  // namespace libria
