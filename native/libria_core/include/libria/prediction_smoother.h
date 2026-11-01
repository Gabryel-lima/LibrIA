#pragma once
// Suavização de probabilidades e supressão de duplicatas.
//
// Referência Python: src/inference/prediction_smoother.py.

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace libria {

// Média móvel das últimas `window_size` distribuições de probabilidade.
class ProbabilitySmoother {
public:
    explicit ProbabilitySmoother(std::size_t window_size = 5);  // > 0

    std::size_t size() const noexcept { return count_; }
    void reset() noexcept;

    // Lança std::invalid_argument se vazia ou de tamanho diferente das anteriores.
    void update(const std::vector<double>& probabilities);

    // Índice e confiança da classe mais provável na média (primeira em empate).
    std::optional<std::pair<std::size_t, double>> best() const;

private:
    std::size_t window_size_;
    std::size_t classes_ = 0;
    std::vector<double> window_;  // window_size x classes, circular
    std::size_t head_ = 0;
    std::size_t count_ = 0;
};

// Bloqueia repetições do mesmo rótulo dentro de uma janela de tempo. Um sinal
// sustentado renova a janela a cada tentativa suprimida.
class DuplicateSuppressor {
public:
    explicit DuplicateSuppressor(double window_seconds = 1.0);  // >= 0

    void reset() noexcept;
    bool should_emit(const std::string& label, double timestamp) const;
    void register_emission(const std::string& label, double timestamp);
    bool accept(const std::string& label, double timestamp);

private:
    double window_seconds_;
    std::optional<std::string> last_label_;
    double last_timestamp_ = 0.0;
};

}  // namespace libria
