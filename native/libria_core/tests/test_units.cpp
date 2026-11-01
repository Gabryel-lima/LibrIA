// Testes de unidade que não dependem das fixtures: contratos de entrada e
// casos de borda da API C++.

#include <stdexcept>
#include <vector>

#include "libria/motion_detector.h"
#include "libria/prediction_smoother.h"
#include "libria/sign_segmenter.h"
#include "libria/sign_token.h"
#include "libria/temporal_buffer.h"
#include "libria/temporal_pipeline.h"
#include "test_framework.h"

namespace {
template <typename Function>
bool throws_invalid_argument(Function function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

void noop_predictor(const float*, std::size_t, std::size_t, std::vector<double>& out) { out.assign(2, 0.5); }
}  // namespace

TEST_CASE(invalid_arguments_are_rejected) {
    EXPECT(throws_invalid_argument([] { libria::MotionDetector(1.0); }), "smoothing = 1");
    EXPECT(throws_invalid_argument([] { libria::MotionDetector(-0.1); }), "smoothing < 0");
    EXPECT(throws_invalid_argument([] { libria::MotionDetector(0.5, 0); }), "coordinates_per_point = 0");
    EXPECT(throws_invalid_argument([] { libria::TemporalBuffer(0, 6); }), "capacity = 0");
    EXPECT(throws_invalid_argument([] { libria::ProbabilitySmoother(0); }), "window_size = 0");
    EXPECT(throws_invalid_argument([] { libria::DuplicateSuppressor(-1.0); }), "window < 0");
    EXPECT(throws_invalid_argument([] {
        libria::SegmenterConfig config;
        config.end_threshold = config.start_threshold * 2;
        libria::SignSegmenter(config, 6);
    }), "histerese invertida");
    EXPECT(throws_invalid_argument([] {
        std::vector<std::size_t> out;
        libria::resample_indices(0, 30, out);
    }), "sequência vazia");
    EXPECT(throws_invalid_argument([] {
        libria::PipelineConfig config;
        config.partial_interval_frames = 0;
        libria::TemporalPipeline(noop_predictor, {}, 30, 6, nullptr, config);
    }), "partial_interval_frames = 0 (no Python seria divisão por zero)");
    EXPECT(throws_invalid_argument([] { libria::TemporalPipeline(nullptr, {}, 30, 6); }), "sem preditor temporal");
}

TEST_CASE(empty_distribution_is_an_error) {
    auto empty = [](const float*, std::size_t, std::size_t, std::vector<double>& out) { out.clear(); };
    libria::PipelineConfig config;
    config.partial_interval_frames = 1;
    libria::TemporalPipeline pipeline(empty, {{0, "J"}}, 2, 3, nullptr, config);
    bool threw = false;
    float frame[3] = {0.0f, 0.0f, 0.0f};
    try {
        for (int i = 0; i < 20 && !threw; ++i) {
            frame[0] += 0.05f;  // movimento contínuo para abrir um segmento
            pipeline.process_frame(frame, i / 30.0);
        }
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT(threw, "distribuição vazia deveria lançar std::runtime_error");
}

TEST_CASE(buffer_window_keeps_order) {
    libria::TemporalBuffer buffer(3, 1);
    float out[3] = {};
    EXPECT(!buffer.window(1, out), "buffer vazio não tem janela");
    for (int i = 0; i < 5; ++i) {
        const float value = static_cast<float>(i);
        buffer.append(&value, i);
    }
    EXPECT(buffer.window(3, out) && out[0] == 2 && out[1] == 3 && out[2] == 4, "janela circular fora de ordem");
    EXPECT(buffer.window(2, out) && out[0] == 3 && out[1] == 4, "janela parcial");
    EXPECT(!buffer.window(4, out), "janela maior que a capacidade");
}

TEST_CASE(normalize_label_strips_and_uppercases_ascii) {
    EXPECT(libria::normalize_label("  oi \n") == "OI");
    EXPECT(libria::normalize_label("j") == "J");
    EXPECT(libria::normalize_label("") == "");
}

TEST_CASE(pipeline_reset_clears_state) {
    libria::TemporalPipeline pipeline(noop_predictor, {{0, "J"}}, 30, 3);
    float frame[3] = {0.1f, 0.2f, 0.3f};
    for (int i = 0; i < 10; ++i) {
        frame[0] += 0.05f;
        pipeline.process_frame(frame, i / 30.0);
    }
    EXPECT(pipeline.is_signing(), "deveria estar sinalizando após movimento contínuo");
    pipeline.reset();
    EXPECT(!pipeline.is_signing() && pipeline.last_energy() == 0.0 && !pipeline.last_token().has_value());
}
