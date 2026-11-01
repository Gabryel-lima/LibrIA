#pragma once
// Harness de teste mínimo (sem dependências externas).

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace testing {

struct State {
    std::vector<std::pair<std::string, std::function<void()>>> tests;
    std::string current;
    int failures_in_test = 0;
    int total_failures = 0;
    std::string fixture_path;
};

inline State& state() {
    static State instance;
    return instance;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> body) {
        state().tests.emplace_back(name, std::move(body));
    }
};

inline void fail(const char* file, int line, const char* expression, const std::string& message) {
    State& s = state();
    ++s.failures_in_test;
    ++s.total_failures;
    if (s.failures_in_test <= 15) {
        std::cerr << "  FALHA [" << s.current << "] " << file << ":" << line << ": " << expression;
        if (!message.empty()) std::cerr << " — " << message;
        std::cerr << "\n";
    } else if (s.failures_in_test == 16) {
        std::cerr << "  ... (mais falhas omitidas neste teste)\n";
    }
}

}  // namespace testing

#define TEST_CASE(name)                                                  \
    static void name();                                                  \
    static ::testing::Registrar registrar_##name(#name, name);           \
    static void name()

// EXPECT(condição) ou EXPECT(condição, mensagem << com << operador <<).
// Os dois "" extras garantem argumentos para o "..." sem extensões GNU.
#define EXPECT(...) LIBRIA_EXPECT_IMPL(__VA_ARGS__, "", "")
#define LIBRIA_EXPECT_IMPL(condition, message, ...)                     \
    do {                                                                 \
        if (!(condition)) {                                              \
            std::ostringstream message_stream_;                          \
            message_stream_ << message;                                  \
            ::testing::fail(__FILE__, __LINE__, #condition, message_stream_.str()); \
        }                                                                \
    } while (0)
