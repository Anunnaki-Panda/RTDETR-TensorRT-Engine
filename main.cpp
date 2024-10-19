#include <iostream>
#include "infer_framework.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <fstream>

// 共享资源
std::mutex mtx;
std::condition_variable cv_;
bool data_ready = false;

// 预处理函数
void preprocess(const std::string &image_path, std::vector<float> &img_vec, cv::Mat &img_resized) {
    auto img_raw = cv::imread(image_path);
    if (img_raw.empty()) {
        std::cerr << "Failed to load image at: " << image_path << std::endl;
        return;
    }

    // 将 BGR 转为 RGB
    cv::Mat img;
    cv::cvtColor(img_raw, img, cv::COLOR_BGR2RGB);

    // 调整尺寸为 640x640
    // cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(640, 640));

    // 归一化处理（将像素值缩放到 [0, 1]）
    cv::Mat img_normalized;
    img_resized.convertTo(img_normalized, CV_32F, 1.0 / 255.0);

    // 将 HWC (Height, Width, Channels) 转为 CHW (Channels, Height, Width)
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < img_normalized.rows; ++h) {
            for (int w = 0; w < img_normalized.cols; ++w) {
                img_vec.push_back(img_normalized.at<cv::Vec3f>(h, w)[c]); // 逐通道存储数据
            }
        }
    }

    // 通知推理线程数据准备就绪
    {
        std::lock_guard<std::mutex> lk(mtx);
        data_ready = true;
    }
    cv_.notify_one(); // 唤醒推理线程
}

// 推理函数
void infer_process(infer_framework &infer, void *input_images_Device, void *input_orig_target_sizes_Device,
                   void *output_output_labels_Device, void *output_box_device, void *output_scores_device,
                   std::vector<long> &output_output_labels_Data, std::vector<float> &output_box_data,
                   std::vector<float> &output_score_data, int output_output_labels_Size,
                   int outout_box_size, int output_scores_size, std::vector<float> &img_vec, int input_images_Size) {
    // 等待数据准备好
    std::unique_lock<std::mutex> lk(mtx);
    cv_.wait(lk, [] {
        return data_ready;
    }); // 等待预处理线程的通知

    // 数据准备好后，开始推理
    cudaMemcpy(input_images_Device, img_vec.data(), input_images_Size * sizeof(float), cudaMemcpyHostToDevice);
    std::ofstream out("input_images_Device.bin", std::ios::binary);
    out.write(reinterpret_cast<char *>(input_images_Device), sizeof(float) * input_images_Size);
    out.close();

    infer.context->setTensorAddress("images", input_images_Device);
    infer.context->setTensorAddress("orig_target_sizes", input_orig_target_sizes_Device);
    infer.context->setTensorAddress("labels", output_output_labels_Device);
    infer.context->setTensorAddress("boxes", output_box_device);
    infer.context->setTensorAddress("scores", output_scores_device);

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Inferencing...\n";
    infer.context->enqueueV3(0);
    cudaStreamSynchronize(0); // 等待推理完成

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_span = t2 - t1;
    std::cout << "Inference took " << time_span.count() << " milliseconds.\n";

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // 拷贝输出数据回 Host
    cudaMemcpy(output_output_labels_Data.data(), output_output_labels_Device, output_output_labels_Size * sizeof(long),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(output_box_data.data(), output_box_device, outout_box_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_score_data.data(), output_scores_device, output_scores_size * sizeof(float),
               cudaMemcpyDeviceToHost);
}

int main() {
    infer_framework infer;
    auto y = infer.engine->getNbIOTensors();
    for (auto i = 0; i < y; i++) {
        auto x = infer.engine->getIOTensorName(i);
        std::cout << std::string(x) << std::endl;
    }


    const char *input_images_Name = "images";
    const char *input_orig_target_sizes_Name = "orig_target_sizes";
    const char *output_labels_Name = "labels";
    const char *output_boxes_Name = "boxes";
    const char *output_scores_Name = "scores";


    std::vector<float> img_vec;

    const int batchSize = 1;

    nvinfer1::Dims input_images_Dims = infer.engine->getTensorShape(input_images_Name);
    const int input_images_Size = batchSize * input_images_Dims.d[0] * input_images_Dims.d[1] *
                                  input_images_Dims.d[2] * input_images_Dims.d[3];

    void *input_images_Device;
    cudaMallocManaged(&input_images_Device, input_images_Size * sizeof(float));

    nvinfer1::Dims input_orig_target_sizes_Dims = infer.engine->getTensorShape(input_orig_target_sizes_Name);
    const int input_orig_target_sizes_Size = batchSize * input_orig_target_sizes_Dims.d[0] *
                                             input_orig_target_sizes_Dims.d[1];

    std::vector<long> input_orig_target_sizes_Data(input_orig_target_sizes_Size, 640);
    std::ofstream out1("input_orig_target_sizes_Device.bin", std::ios::binary);
    out1.write(reinterpret_cast<char *>(input_orig_target_sizes_Data.data()),
               sizeof(long) * input_orig_target_sizes_Size);
    out1.close();

    void *input_orig_target_sizes_Device;
    cudaMallocManaged(&input_orig_target_sizes_Device, input_orig_target_sizes_Size * sizeof(long));
    cudaMemcpy(input_orig_target_sizes_Device, input_orig_target_sizes_Data.data(),
               input_orig_target_sizes_Size * sizeof(long), cudaMemcpyHostToDevice);

    nvinfer1::Dims output_labels_Dims = infer.engine->getTensorShape(output_labels_Name);
    const int output_output_labels_Size = batchSize * output_labels_Dims.d[1];
    std::vector<long> output_output_labels_Data(output_output_labels_Size);
    void *output_output_labels_Device;
    cudaMallocManaged(&output_output_labels_Device, output_output_labels_Size * sizeof(long));

    nvinfer1::Dims output_boxes_Dims = infer.engine->getTensorShape(output_boxes_Name);
    const int output_box_size = batchSize * output_boxes_Dims.d[1] * output_boxes_Dims.d[2];
    std::vector<float> output_box_data(output_box_size);
    void *output_box_device;
    cudaMallocManaged(&output_box_device, output_box_size * sizeof(float));

    nvinfer1::Dims output_scores_Dims = infer.engine->getTensorShape(output_scores_Name);
    const int output_scores_size = batchSize * output_scores_Dims.d[1];
    std::vector<float> output_score_data(output_scores_size);
    void *output_scores_device;
    cudaMallocManaged(&output_scores_device, output_scores_size * sizeof(float));

    cv::Mat img_resized;
    std::thread preprocess_thread(preprocess, "img.png", std::ref(img_vec),
                                  std::ref(img_resized));

    // 启动推理线程
    std::thread infer_thread(infer_process, std::ref(infer), input_images_Device, input_orig_target_sizes_Device,
                             output_output_labels_Device, output_box_device, output_scores_device,
                             std::ref(output_output_labels_Data), std::ref(output_box_data),
                             std::ref(output_score_data),
                             output_output_labels_Size, output_box_size, output_scores_size, std::ref(img_vec),
                             input_images_Size);

    preprocess_thread.join();
    infer_thread.join();

    for (size_t i = 0; i < output_score_data.size(); i++) {
        const auto &score = output_score_data[i];
        if (score > 0.8) {
            const auto &x_min = output_box_data[i * 4];
            const auto &y_min = output_box_data[i * 4 + 1];
            const auto &x_max = output_box_data[i * 4 + 2];
            const auto &y_max = output_box_data[i * 4 + 3];

            cv::rectangle(img_resized,
                          cv::Point(static_cast<int>(x_min), static_cast<int>(y_min)),
                          cv::Point(static_cast<int>(x_max), static_cast<int>(y_max)),
                          cv::Scalar(0, 255, 0), 2); // 绿色框，线条宽度为2

            const auto &cls = output_output_labels_Data[i];
            std::string label = "Class: " + std::to_string(cls) + ", Score: " + std::to_string(score);

            int text_x = static_cast<int>(x_min);
            int text_y = static_cast<int>(y_min) - 5; // 防止文本覆盖在框上，稍微向上偏移

            cv::putText(img_resized, label, cv::Point(text_x, text_y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1); // 绿色文本，字体大小0.5，宽度1
        }
    }


    cv::imshow("preprocess", img_resized);
    cv::waitKey(0);
    std::cout << "Hello, World!" << std::endl;
    return 0;
}