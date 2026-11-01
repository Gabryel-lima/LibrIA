cmake_minimum_required(VERSION 3.16)
project(libria_core VERSION 0.1.0 LANGUAGES CXX)

# Núcleo nativo do LibrIA: features, detector de movimento, segmentação
# temporal, suavização, regra híbrida e quantização int8. Sem dependências
# externas e sem TFLite: os modelos entram por injeção de dependência.
# Compila no host (testes) e no Android via NDK.

option(LIBRIA_BUILD_TESTS "Compila os testes de paridade e de unidade" ON)
option(LIBRIA_SANITIZE "Compila com AddressSanitizer e UndefinedBehaviorSanitizer" OFF)
option(LIBRIA_WERROR "Trata warnings como erro" OFF)

set(LIBRIA_FIXTURES "${CMAKE_CURRENT_SOURCE_DIR}/../fixtures/parity_fixtures.json"
    CACHE FILEPATH "Fixtures de paridade geradas por scripts/export_parity_fixtures.py")

add_library(libria_core
    src/features.cpp
    src/hybrid.cpp
    src/motion_detector.cpp
    src/numeric.cpp
    src/prediction_smoother.cpp
    src/quantization.cpp
    src/sign_segmenter.cpp
    src/sign_token.cpp
    src/temporal_buffer.cpp
    src/temporal_pipeline.cpp
)
add_library(libria::core ALIAS libria_core)
target_include_directories(libria_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(libria_core PUBLIC cxx_std_17)
set_target_properties(libria_core PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Paridade bit a bit com o NumPy: o compilador não pode fundir multiplicação
    # e soma em FMA (no ARM64 do Android ele faria isso por padrão).
    target_compile_options(libria_core PRIVATE -ffp-contract=off)
    set(LIBRIA_WARNINGS -Wall -Wextra -Wpedantic -Wshadow)
    if(LIBRIA_WERROR)
        list(APPEND LIBRIA_WARNINGS -Werror)
    endif()
    target_compile_options(libria_core PRIVATE ${LIBRIA_WARNINGS})

    if(LIBRIA_SANITIZE)
        set(LIBRIA_SANITIZER_FLAGS -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
        target_compile_options(libria_core PUBLIC ${LIBRIA_SANITIZER_FLAGS})
        target_link_options(libria_core PUBLIC ${LIBRIA_SANITIZER_FLAGS})
    endif()
endif()

if(LIBRIA_BUILD_TESTS)
    enable_testing()
    add_executable(libria_core_tests tests/test_main.cpp tests/test_parity.cpp tests/test_units.cpp)
    target_link_libraries(libria_core_tests PRIVATE libria::core)
    set_target_properties(libria_core_tests PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(libria_core_tests PRIVATE ${LIBRIA_WARNINGS})
    endif()
    add_test(NAME libria_core_tests COMMAND libria_core_tests ${LIBRIA_FIXTURES})
endif()
