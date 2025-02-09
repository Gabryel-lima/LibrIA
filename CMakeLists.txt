# cmake_minimum_required(VERSION 3.10)
# project(LibrIA)

# # Definir padrão C++
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# # Caminho do OpenCV
# set(OpenCV_DIR "~/libs/opencv/install/lib/cmake/opencv4")
# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})

# # Caminho do LibTorch
# set(Torch_DIR "~/Documentos/projects/LibrIA/src/interfaces/dep/libtorch/share/cmake/Torch")
# find_package(Torch REQUIRED)
# include_directories(${TORCH_INCLUDE_DIRS})

# # Criar o executável
# add_executable(cam src/interfaces/cam.cpp)

# # Vincular bibliotecas necessárias
# target_link_libraries(cam ${OpenCV_LIBS} ${TORCH_LIBRARIES} 
#     opencv_core 
#     opencv_highgui 
#     opencv_imgproc 
#     opencv_imgcodecs)

