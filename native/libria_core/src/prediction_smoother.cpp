#include "libria/prediction_smoother.h"

#include <stdexcept>

namespace libria {

ProbabilitySmoother::ProbabilitySmoother(std::size_t window_size) : window_size_(window_size) {
    if (window_size == 0) {
        throw std::invalid_argument("window_size deve ser maior que zero");
    }
}

void ProbabilitySmoother::reset() noexcept {
    head_ = 0;
    count_ = 0;
}

void ProbabilitySmoother::update(const std::vector<double>& probabilities) {
    if (probabilities.empty()) {
        throw std::invalid_argument("probabilities não pode ser vazio");
    }
    if (count_ > 0 && probabilities.size() != classes_) {
        throw std::invalid_argument("Todas as distribuições devem ter o mesmo tamanho");
    }
    if (classes_ != probabilities.size()) {
        classes_ = probabilities.size();
        window_.assign(window_size_ * classes_, 0.0);
        head_ = 0;
        count_ = 0;
    }
    std::copy(probabilities.begin(), probabilities.end(),
              window_.begin() + static_cast<std::ptrdiff_t>(head_ * classes_));
    head_ = (head_ + 1) % window_size_;
    if (count_ < window_size_) {
        ++count_;
    }
}

std::optional<std::pair<std::size_t, double>> ProbabilitySmoother::best() const {
    if (count_ == 0) {
        return std::nullopt;
    }
    // np.mean(np.stack(window), axis=0): soma sequencial do mais antigo ao mais
    // novo, a partir de 0, e divisão pelo número de distribuições.
    const std::size_t oldest = (head_ + window_size_ - count_) % window_size_;
    std::size_t best_index = 0;
    double best_value = 0.0;
    for (std::size_t c = 0; c < classes_; ++c) {
        double sum = 0.0;
        for (std::size_t i = 0; i < count_; ++i) {
            const std::size_t slot = (oldest + i) % window_size_;
            sum += window_[slot * classes_ + c];
        }
        const double average = sum / static_cast<double>(count_);
        if (c == 0 || average > best_value) {  // np.argmax: primeira em empate
            best_index = c;
            best_value = average;
        }
    }
    return std::make_pair(best_index, best_value);
}

DuplicateSuppressor::DuplicateSuppressor(double window_seconds) : window_seconds_(window_seconds) {
    if (window_seconds < 0.0) {
        throw std::invalid_argument("window_seconds não pode ser negativo");
    }
}

void DuplicateSuppressor::reset() noexcept {
    last_label_.reset();
    last_timestamp_ = 0.0;
}

bool DuplicateSuppressor::should_emit(const std::string& label, double timestamp) const {
    if (!last_label_.has_value()) {
        return true;
    }
    if (label != *last_label_) {
        return true;
    }
    return (timestamp - last_timestamp_) >= window_seconds_;
}

void DuplicateSuppressor::register_emission(const std::string& label, double timestamp) {
    last_label_ = label;
    last_timestamp_ = timestamp;
}

bool DuplicateSuppressor::accept(const std::string& label, double timestamp) {
    const bool emit = should_emit(label, timestamp);
    // Um sinal sustentado renova a janela: só volta a emitir depois de
    // window_seconds sem o sinal, e não window_seconds após a 1ª emissão.
    register_emission(label, timestamp);
    return emit;
}

}  // namespace libria
