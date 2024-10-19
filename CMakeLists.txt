cmake_minimum_required(VERSION 3.16.3)
project(RTDETR)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenCV REQUIRED)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.4)
set(CUDA_NVCC_EXECUTABLE /usr/local/cuda-12.4/bin/nvcc)

set(CUDA_CUDART_LIBRARY /usr/local/cuda-12.4/lib64/libcudart.so)

find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

set(TensorRT_DIR /usr/local/TensorRT-10.0.1.6/)
include_directories(${TensorRT_DIR}/include/)
link_directories(${TensorRT_DIR}/lib)

add_executable(${PROJECT_NAME} main.cpp infer_framework.cpp infer_framework.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CUDA_LIBRARIES} nvinfer pthread)