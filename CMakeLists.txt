cmake_minimum_required(VERSION 3.16.3)
project(RTDETR)

set(CMAKE_CXX_STANDARD 20)

# 链接 OpenCV 库和 CUDA 库
target_link_libraries(${OpenCV_LIBS} ${CUDA_LIBRARIES})

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# 添加 TensorRT 路径 (修改为你本地的 TensorRT 安装路径)
include_directories(D:/TensorRT-10.1.0.27/include)
link_directories(D:/TensorRT-10.1.0.27/lib)

include_directories(D:/vcpkg/packages/opencv4_x64-windows/include)
link_directories(D:/vcpkg/packages/opencv4_x64-windows/lib)
add_executable(RTDETR main.cpp
        infer_framework.cpp
        infer_framework.h)
