# RT-DETR TensorRT Engine Deployment

This project demonstrates how to deploy the **RT-DETR** model using **C++** and **TensorRT**. For the model details, please refer to the [RT-DETR GitHub Repository](https://github.com/lyuwenyu/RT-DETR).

## Prerequisites

Before getting started, ensure that the following software versions are installed:

- **TensorRT**: 10.0.1.6
- **OpenCV**: 4.8.1
- **CUDA**: 12.3

## Project Overview

RT-DETR is a high-performance real-time detection model that is deployed using TensorRT to maximize inference efficiency. This project provides a clean and optimized C++ implementation to help users integrate RT-DETR into their own applications.

## Setup

To get started with this project, make sure the required dependencies are installed, and follow the steps below to build and run the deployment engine.

1. Clone the repository:
   ```bash
   git clone https://github.com/Anunnaki-Panda/RTDETR-TensorRT-Engine
   cd RTDETR-TensorRT-Engine

2. Build the project with CMake:
    ```bash
    mkdir build && cd build
    cmake ..
    make
3. Run the deployment:
    ```bash
   ./RTDETR

