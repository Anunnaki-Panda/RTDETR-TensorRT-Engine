#ifndef INFER_FRAMEWORK_H
#define INFER_FRAMEWORK_H

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <vector>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity > Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
};

class infer_framework {
public:
    Logger logger;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    const std::string engineFile = "./model/model.trt";
    nvinfer1::ICudaEngine* engine = loadEngine(engineFile, runtime);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    nvinfer1::ICudaEngine* loadEngine(const std::string& engineFile, nvinfer1::IRuntime* runtime) {
        std::ifstream file(engineFile, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open engine file: " << engineFile << std::endl;
            return nullptr;
        }

        file.seekg(0, std::ifstream::end);
        size_t size = file.tellg();
        file.seekg(0, std::ifstream::beg);

        std::vector<char> buffer(size);
        file.read(buffer.data(), size);

        return runtime->deserializeCudaEngine(buffer.data(), size);
    }
};

#endif //INFER_FRAMEWORK_H
