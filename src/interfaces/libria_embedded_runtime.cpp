#include "libria_embedded_runtime.h"

LibriaEmbeddedRuntime::LibriaEmbeddedRuntime() = default;

bool LibriaEmbeddedRuntime::Init(
    const unsigned char* static_model_data,
    std::size_t static_model_size,
    const unsigned char* temporal_model_data,
    std::size_t temporal_model_size,
    std::uint8_t* tensor_arena,
    std::size_t tensor_arena_size
) {
    (void)static_model_data;
    (void)static_model_size;
    (void)temporal_model_data;
    (void)temporal_model_size;
    (void)tensor_arena;
    (void)tensor_arena_size;

    // TODO: connect TFLite Micro interpreters here.
    // The generated bundle header in model/embedded_bundle/ exposes the model
    // shapes, thresholds and class labels that should stay aligned with the
    // host-side training/export pipeline.
    return true;
}

LibriaEmbeddedPrediction LibriaEmbeddedRuntime::PredictStatic(const float landmarks[21][3]) {
    (void)landmarks;
    // TODO: run static TFLite Micro interpreter and decode the top-1 class.
    return MakeInvalidPrediction();
}

LibriaEmbeddedPrediction LibriaEmbeddedRuntime::PredictTemporal(const float sequence[30][63]) {
    (void)sequence;
    // TODO: run temporal TFLite Micro interpreter and decode the top-1 class.
    return MakeInvalidPrediction();
}

LibriaEmbeddedPrediction LibriaEmbeddedRuntime::PredictHybrid(
    const float landmarks[21][3],
    const float sequence[30][63],
    bool has_temporal_window
) {
    LibriaEmbeddedPrediction static_prediction = PredictStatic(landmarks);
    if (!has_temporal_window) {
        return static_prediction;
    }

    LibriaEmbeddedPrediction temporal_prediction = PredictTemporal(sequence);
    if (temporal_prediction.valid) {
        return temporal_prediction;
    }

    return static_prediction;
}

LibriaEmbeddedPrediction LibriaEmbeddedRuntime::MakeInvalidPrediction() const {
    return {nullptr, 0.0f, false};
}