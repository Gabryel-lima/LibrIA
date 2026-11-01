#pragma once
// Saída padronizada do reconhecimento.
//
// Referência Python: src/inference/sign_token.py, SignToken.
// O tipo de sinal e a tradução (vocabulário) ficam fora do núcleo: quem consome
// o token resolve pelo rótulo.

#include <string>

namespace libria {

enum class TokenState { kPartial, kFinal, kRejected };
enum class TokenSource { kStatic, kTemporal };

const char* to_string(TokenState state) noexcept;
const char* to_string(TokenSource source) noexcept;

struct SignToken {
    std::string label;
    double confidence = 0.0;
    double start_time = 0.0;
    double end_time = 0.0;
    TokenSource source = TokenSource::kStatic;
    TokenState state = TokenState::kFinal;
    int frame_count = 0;

    double duration_seconds() const noexcept {
        return end_time > start_time ? end_time - start_time : 0.0;
    }
};

// Normalização de rótulo do Python (str.strip().upper()), restrita a ASCII:
// os rótulos do vocabulário são ASCII (acentos só aparecem nas traduções).
std::string normalize_label(const std::string& label);

}  // namespace libria
