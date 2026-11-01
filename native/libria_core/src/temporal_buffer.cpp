#include "libria/temporal_buffer.h"

#include <algorithm>
#include <stdexcept>

#include "libria/numeric.h"

namespace libria {

void resample_indices(std::size_t frames, std::size_t target, std::vector<std::size_t>& out) {
    if (target == 0) {
        throw std::invalid_argument("target_length deve ser maior que zero");
    }
    if (frames == 0) {
        throw std::invalid_argument("Não é possível reamostrar uma sequência vazia");
    }
    out.resize(target);
    if (frames == target) {
        for (std::size_t i = 0; i < target; ++i) {
            out[i] = i;
        }
        return;
    }
    // np.linspace(0, frames - 1, target): y = i * step (step = delta / div),
    // e o último ponto é exatamente `stop`. Com target == 1 o passo é
    // indefinido e o NumPy devolve [0].
    if (target == 1) {
        out[0] = 0;
        return;
    }
    const double stop = static_cast<double>(frames - 1);
    const double step = stop / static_cast<double>(target - 1);
    for (std::size_t i = 0; i < target; ++i) {
        const double y = (i + 1 == target) ? stop : static_cast<double>(i) * step;
        out[i] = static_cast<std::size_t>(numeric::round_half_even(y));
    }
}

void resample_sequence(const float* frames, std::size_t frame_count, std::size_t feature_dim,
                       std::size_t target, std::vector<std::size_t>& index_scratch, float* out) {
    resample_indices(frame_count, target, index_scratch);
    for (std::size_t i = 0; i < target; ++i) {
        const float* source = frames + index_scratch[i] * feature_dim;
        std::copy(source, source + feature_dim, out + i * feature_dim);
    }
}

TemporalBuffer::TemporalBuffer(std::size_t capacity, std::size_t feature_dim)
    : capacity_(capacity), feature_dim_(feature_dim) {
    if (capacity == 0) {
        throw std::invalid_argument("capacity deve ser maior que zero");
    }
    if (feature_dim == 0) {
        throw std::invalid_argument("feature_dim deve ser maior que zero");
    }
    frames_.resize(capacity * feature_dim);
    timestamps_.resize(capacity);
}

void TemporalBuffer::append(const float* features, double timestamp) noexcept {
    std::copy(features, features + feature_dim_, frames_.begin() + static_cast<std::ptrdiff_t>(head_ * feature_dim_));
    timestamps_[head_] = timestamp;
    head_ = (head_ + 1) % capacity_;
    if (size_ < capacity_) {
        ++size_;
    }
}

void TemporalBuffer::reset() noexcept {
    head_ = 0;
    size_ = 0;
}

bool TemporalBuffer::window(std::size_t length, float* out) const noexcept {
    if (length == 0 || length > size_) {
        return false;
    }
    // O mais antigo da janela está `length` posições antes de head_.
    std::size_t slot = (head_ + capacity_ - length) % capacity_;
    for (std::size_t i = 0; i < length; ++i) {
        const float* source = frames_.data() + slot * feature_dim_;
        std::copy(source, source + feature_dim_, out + i * feature_dim_);
        slot = (slot + 1) % capacity_;
    }
    return true;
}

}  // namespace libria
