#include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <vector>

#include "config/config.hpp"
#include "cuda_utils.h"
#include "logging/logging.h"
#include "pixel_shuffle/pixel_shuffle.hpp"
#include "preprocess/preprocess.hpp"

static Logger gLogger;

using namespace nvinfer1;

// TensorRT weight files
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. Please check if the .wts file path is correct.");

    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(uint32_t) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

auto* ConvPRelu(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int conv_nb, int index) {
    IConvolutionLayer* conv = network->addConvolutionNd(input, conv_nb, DimsHW{3, 3},
                                                        weightMap["body." + std::to_string(index) + ".weight"],
                                                        weightMap["body." + std::to_string(index) + ".bias"]);
    assert(conv);
    conv->setStrideNd(DimsHW{1, 1});
    conv->setPaddingNd(DimsHW{1, 1});
    
    auto slope = network->addConstant(Dims4{1, 64, 1, 1}, weightMap["body." + std::to_string(index + 1) + ".weight"]);
    assert(slope);

    auto prelu = network->addParametricReLU(*conv->getOutput(0), *slope->getOutput(0));
    assert(prelu);

    return prelu;
}

void build_engine(DataType dt, std::string& wts_path) {
    std::map<std::string, Weights> weightMap = loadWeights(wts_path);

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U);

    auto data = network->addInput(INPUT_BLOB_NAME, nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{BATCH_SIZE, INPUT_C, INPUT_H, INPUT_W});

    auto layer = ConvPRelu(network, weightMap, *data, 64, 0);
    for (int i = 0; i < 32; ++i) {
        layer = ConvPRelu(network, weightMap, *layer->getOutput(0), 64, 2 * i + 2);
    }

    auto conv_last = network->addConvolutionNd(*layer->getOutput(0), 48, DimsHW{3, 3}, weightMap["body.66.weight"], weightMap["body.66.bias"]);
    assert(conv_last);
    auto conv_last_res = conv_last->getOutput(0);

    IPluginCreator* creator = getPluginRegistry()->getPluginCreator("PixelShufflePlugin", "1");
    int upscaleFactor = 4;
    std::vector<PluginField> pluginData = {PluginField{"upscaleFactor", &upscaleFactor, PluginFieldType::kINT32, 1}};
    PluginFieldCollection pluginFCWithData = {static_cast<int>(pluginData.size()), pluginData.data()};
    auto pluginObj = creator->createPlugin("PixelShuffle", &pluginFCWithData);

    auto pixelShuffleLayer = network->addPluginV2(&conv_last_res, 1, *pluginObj);
    auto interpolateLayer = network->addResize(*data);
    interpolateLayer->setResizeMode(ResizeMode::kNEAREST);
    // float scales[] = {1.0f, 1.0f, 1.0 * OUT_SCALE, 1.0 * OUT_SCALE};
    // interpolateLayer->setScales(scales, OUT_SCALE);

    interpolateLayer->setOutputDimensions(pixelShuffleLayer->getOutput(0)->getDimensions());


    auto addLayer = network->addElementWise(*interpolateLayer->getOutput(0), *pixelShuffleLayer->getOutput(0), ElementWiseOperation::kSUM);
    addLayer->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*addLayer->getOutput(0));

    if (USE_FP16) {
        config->setFlag(BuilderFlag::kFP16);
    }

    std::cout << "Building engine, please wait for a while..." << std::endl;
    IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    std::ofstream ofs("../weights/real-esrgan.engine", std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(serialized_model->data()), serialized_model->size());

    delete config;
    delete serialized_model;
    delete builder;
}

// Function to handle inference
void doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output) {
    context.setBindingDimensions(0, Dims4(BATCH_SIZE, INPUT_C, INPUT_H, INPUT_W));
    context.enqueueV2(buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <output_video_path>" << std::endl;
        return -1;
    }
    
    std::string output_video_path = argv[1];
    std::string wts_path = "../weights/real-esrgan.wts";
    build_engine(DataType::kFLOAT, wts_path);

    std::string engine_name = "../weights/real-esrgan.engine";
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: Could not read the engine file!" << std::endl;
        return -1;
    }

    char* trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    file.read(trtModelStream, size);
    file.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    IExecutionContext* context = engine->createExecutionContext();
    delete[] trtModelStream;

    void* buffers[2];
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    std::vector<float> data(BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W);
    std::vector<float> output(BATCH_SIZE * OUTPUT_SIZE);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cv::VideoCapture cap("/home/ubuntu-test/tuan anh/realesrganv3/VID_IR_0.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video stream!" << std::endl;
        return -1;
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

    cv::VideoWriter writer(output_video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, 
                       cv::Size(frame_width * OUT_SCALE, frame_height * OUT_SCALE));


    if (!writer.isOpened()) {
    std::cerr << "Error: Could not open the video writer!" << std::endl;
    return -1;
    }   

    cv::Mat frame;
    int frame_count = 0;  // Đếm số thứ tự của khung hình
    while (cap.read(frame)) {
        if (frame.empty()) {
            std::cerr << "Error: Could not read frame!" << std::endl;
            break;
        }

        // Bắt đầu đo thời gian cho một khung hình
        auto start = std::chrono::high_resolution_clock::now();

        int i = 0;
        for (int row = 0; row < INPUT_H; ++row) {
            uchar* uc_pixel = frame.data + row * frame.step;
            for (int col = 0; col < INPUT_W; ++col) {
                data[i] = (float)uc_pixel[2] / 255.0;
                data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                uc_pixel += 3;
                ++i;
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], data.data(), BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        doInference(*context, stream, (void**)buffers, output.data());
                // Đo thời gian kết thúc cho khung hình này
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // Tính toán FPS và in ra console
        // double fps = 1.0 / elapsed.count();
        // std::cout << "Processed Frame " << ++frame_count << " - FPS: " << fps << std::endl;
        std::cout << "Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << " ms" << std::endl;
        cv::Mat img_res(INPUT_H * OUT_SCALE, INPUT_W * OUT_SCALE, CV_8UC3);
        i = 0;
        for (int row = 0; row < img_res.rows; ++row) {
            uchar* uc_pixel = img_res.data + row * img_res.step;
            for (int col = 0; col < img_res.cols; ++col) {
                auto r2 = std::round(output[i] * 255.0);
                auto g2 = std::round(output[i + 1 * img_res.rows * img_res.cols] * 255.0);
                auto b2 = std::round(output[i + 2 * img_res.rows * img_res.cols] * 255.0);
                uc_pixel[0] = static_cast<uchar>(std::clamp(b2, 0.0, 255.0));
                uc_pixel[1] = static_cast<uchar>(std::clamp(g2, 0.0, 255.0));
                uc_pixel[2] = static_cast<uchar>(std::clamp(r2, 0.0, 255.0));

                uc_pixel += 3;
                ++i;
            }
        }

        // Ghi khung hình vào video đầu ra
        writer.write(img_res);



    }

    cap.release();
    writer.release();
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    delete context;
    delete engine;
    delete runtime;
    // delete builder;
    return 0;
}
