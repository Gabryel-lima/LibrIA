#pragma once
// Energia de movimento suavizada, quadro a quadro.
//
// Referência Python: src/inference/motion_detector.py, MotionDetector.

#include <cstddef>
#include <vector>

namespace libria {

class MotionDetector {
public:
    // smoothing em [0, 1); coordinates_per_point > 0. Lança std::invalid_argument.
    explicit MotionDetector(double smoothing = 0.5, std::size_t coordinates_per_point = 3);

    double energy() const noexcept { return energy_; }

    // Só a partir do segundo quadro consecutivo com a mão presente a energia é
    // uma medida real (um quadro isolado não tem movimento por definição).
    bool has_reading() const noexcept { return consecutive_frames_ >= 2; }

    void reset() noexcept;

    // `features == nullptr` significa mão ausente: a energia decai em vez de
    // gerar um pico artificial quando a mão reaparece em outra posição.
    // Devolve a energia suavizada.
    double update(const float* features, std::size_t size);

private:
    double raw_energy(const float* features, std::size_t size);

    double smoothing_;
    std::size_t coordinates_per_point_;
    std::vector<float> previous_;
    std::vector<float> scratch_;
    bool has_previous_ = false;
    double energy_ = 0.0;
    int consecutive_frames_ = 0;
};

}  // namespace libria
