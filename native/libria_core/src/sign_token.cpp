#include "libria/sign_token.h"

namespace libria {

const char* to_string(TokenState state) noexcept {
    switch (state) {
        case TokenState::kPartial: return "partial";
        case TokenState::kFinal: return "final";
        case TokenState::kRejected: return "rejected";
    }
    return "unknown";
}

const char* to_string(TokenSource source) noexcept {
    switch (source) {
        case TokenSource::kStatic: return "static";
        case TokenSource::kTemporal: return "temporal";
    }
    return "unknown";
}

namespace {
bool is_ascii_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}
}  // namespace

std::string normalize_label(const std::string& label) {
    std::size_t begin = 0;
    std::size_t end = label.size();
    while (begin < end && is_ascii_space(label[begin])) ++begin;
    while (end > begin && is_ascii_space(label[end - 1])) --end;
    std::string result = label.substr(begin, end - begin);
    for (char& c : result) {
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    }
    return result;
}

}  // namespace libria
