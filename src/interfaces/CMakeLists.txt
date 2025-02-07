cmake_minimum_required(VERSION 3.10)

project(LibrIA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Caminho para o LibTorch
set(Torch_DIR ~/libtorch)
find_package(Torch REQUIRED)

# Caminho para o OpenCV
find_package(OpenCV REQUIRED)

# Arquivo fonte do projeto
add_executable(cam src/interfaces/cam.cpp)

# Incluir diretórios do OpenCV e LibTorch
target_include_directories(cam PRIVATE ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

# Ligar bibliotecas necessárias
target_link_libraries(cam ${OpenCV_LIBS} ${TORCH_LIBRARIES})
