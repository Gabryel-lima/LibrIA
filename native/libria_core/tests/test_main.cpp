#include <exception>
#include <iostream>

#include "test_framework.h"

int main(int argc, char** argv) {
    testing::State& s = testing::state();
    if (argc > 1) s.fixture_path = argv[1];

    int failed_tests = 0;
    for (const auto& test : s.tests) {
        s.current = test.first;
        s.failures_in_test = 0;
        try {
            test.second();
        } catch (const std::exception& error) {
            testing::fail(__FILE__, __LINE__, "exceção", error.what());
        }
        std::cout << (s.failures_in_test == 0 ? "[ OK ] " : "[FAIL] ") << test.first << "\n";
        if (s.failures_in_test != 0) ++failed_tests;
    }
    std::cout << "\n" << (s.tests.size() - static_cast<std::size_t>(failed_tests)) << "/" << s.tests.size()
              << " testes passaram\n";
    return failed_tests == 0 ? 0 : 1;
}
