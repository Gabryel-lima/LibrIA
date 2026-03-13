#pragma once

#include <cstddef>
#include <cstdint>

struct LibriaEmbeddedPrediction {
    const char* token;
    float confidence;
    bool valid;
};

class LibriaEmbeddedRuntime {
public:
    LibriaEmbeddedRuntime();

    // Load the two quantized models and prepare the TFLite Micro interpreters.
    bool Init(
        const unsigned char* static_model_data,
        std::size_t static_model_size,
        const unsigned char* temporal_model_data,
        std::size_t temporal_model_size,
        std::uint8_t* tensor_arena,
        std::size_t tensor_arena_size
    );

    // The landmark contract matches the host pipeline output shapes.
    LibriaEmbeddedPrediction PredictStatic(const float landmarks[21][3]);
    LibriaEmbeddedPrediction PredictTemporal(const float sequence[30][63]);

    // Hybrid rule: temporal J/Z overrides static when above threshold.
    LibriaEmbeddedPrediction PredictHybrid(
        const float landmarks[21][3],
        const float sequence[30][63],
        bool has_temporal_window
    );

private:
    LibriaEmbeddedPrediction MakeInvalidPrediction() const;
};