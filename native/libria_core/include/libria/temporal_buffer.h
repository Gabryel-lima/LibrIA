#pragma once
// Buffer circular de features com carimbo de tempo, e reamostragem.
//
// Referência Python: src/inference/temporal_buffer.py.

#include <cstddef>
#include <vector>

namespace libria {

// Índices escolhidos por resample_sequence(frames, target) no Python:
// vizinho mais próximo sobre np.linspace(0, frames - 1, target), com np.rint.
// Lança std::invalid_argument se target == 0 ou frames == 0.
void resample_indices(std::size_t frames, std::size_t target, std::vector<std::size_t>& out);

// Copia para `out` (target x feature_dim) os quadros escolhidos de `frames`
// (frame_count x feature_dim, contíguos).
void resample_sequence(const float* frames, std::size_t frame_count, std::size_t feature_dim,
                       std::size_t target, std::vector<std::size_t>& index_scratch, float* out);

class TemporalBuffer {
public:
    // capacity > 0 e feature_dim > 0. Lança std::invalid_argument.
    TemporalBuffer(std::size_t capacity, std::size_t feature_dim);

    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return capacity_; }
    bool is_full() const noexcept { return size_ >= capacity_; }

    void append(const float* features, double timestamp) noexcept;
    void reset() noexcept;

    // Copia os últimos `length` quadros, do mais antigo ao mais novo, para `out`
    // (length x feature_dim). Devolve false se não houver tantos quadros.
    bool window(std::size_t length, float* out) const noexcept;

private:
    std::size_t capacity_;
    std::size_t feature_dim_;
    std::vector<float> frames_;
    std::vector<double> timestamps_;
    std::size_t head_ = 0;  // próxima posição de escrita
    std::size_t size_ = 0;
};

}  // namespace libria
