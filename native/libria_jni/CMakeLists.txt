cmake_minimum_required(VERSION 3.18)
project(libria_jni LANGUAGES CXX)

# Ponte JNI entre o Kotlin (io.github.gabryellima.libria.bridge) e o núcleo C++.
# O mesmo código compila para o Android (NDK, via externalNativeBuild do app) e
# para o host (JDK), onde os testes Kotlin do módulo libria-bridge o carregam.

option(LIBRIA_WERROR "Trata warnings como erro" OFF)

set(LIBRIA_BUILD_TESTS OFF CACHE BOOL "Testes do núcleo ficam no build próprio dele" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../libria_core ${CMAKE_CURRENT_BINARY_DIR}/libria_core)

add_library(libria_jni SHARED src/libria_jni.cpp)
target_link_libraries(libria_jni PRIVATE libria::core)
target_compile_features(libria_jni PRIVATE cxx_std_17)
set_target_properties(libria_jni PROPERTIES CXX_EXTENSIONS OFF CXX_VISIBILITY_PRESET hidden)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(LIBRIA_JNI_WARNINGS -Wall -Wextra -Wpedantic -Wshadow)
    if(LIBRIA_WERROR)
        list(APPEND LIBRIA_JNI_WARNINGS -Werror)
    endif()
    target_compile_options(libria_jni PRIVATE -ffp-contract=off ${LIBRIA_JNI_WARNINGS})
endif()

if(ANDROID)
    target_link_libraries(libria_jni PRIVATE log)
    # Aparelhos com páginas de 16 KB (exigência do Android 15+ na Play Store).
    target_link_options(libria_jni PRIVATE "-Wl,-z,max-page-size=16384")
else()
    find_package(JNI REQUIRED)
    target_include_directories(libria_jni PRIVATE ${JNI_INCLUDE_DIRS})
endif()
