/*
  Claire Liu, Yu-Jing Wei
  extractor.cpp

  Path: src/utils/extractor.cpp
  Description: Extracts features from images.
*/

#include "extractor.hpp"
#include "preProcessor.hpp"
#include "regionDetect.hpp"
#include "regionAnalyzer.hpp"
#include "thresholding.hpp"
#include "morphologicalFilter.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#if defined(ENABLE_ONNXRUNTIME)
#include <onnxruntime/onnxruntime_cxx_api.h>
#endif

int BaselineExtractor::extractRegion(
    const RegionFeatures &region,
    std::vector<float> *featureVector) const
{
    if (!featureVector)
    {
        return -1;
    }

    const std::vector<double> shape = getShapeFeatureVector(region);
    featureVector->clear();
    featureVector->reserve(shape.size());
    for (double v : shape)
    {
        featureVector->push_back(static_cast<float>(v));
    }
    return featureVector->empty() ? -1 : 0;
}

int BaselineExtractor::extractMat(
    const cv::Mat &image,
    std::vector<float> *featureVector) const
{
    if (!featureVector || image.empty())
    {
        return -1;
    }

    cv::Mat pre = PreProcessor::imgPreProcess(image, 0.5f, 50, 5);
    cv::Mat binary;
    Threadsholding::dynamicThreadsHold(pre, binary);

    MorphologicalFilter mf;
    cv::Mat cleaned;
    mf.defaultDilationErosion(binary, cleaned);

    cv::Mat labels;
    RegionDetect::twoPassSegmentation(cleaned, labels);

    const int frameArea = image.rows * image.cols;
    const int minAreaPixels = std::max(500, frameArea / 20); // ~5% of frame
    RegionAnalyzer analyzer(RegionAnalyzer::Params(false, minAreaPixels, true));
    auto regions = analyzer.analyzeLabels(labels);
    if (regions.empty())
    {
        return -1;
    }

    auto best = std::max_element(
        regions.begin(), regions.end(),
        [](const RegionFeatures &a, const RegionFeatures &b)
        {
            return a.area < b.area;
        });
    if (best == regions.end())
        return -1;

    return extractRegion(*best, featureVector);
}

int CNNExtractor::extractMat(
    const cv::Mat &image,
    std::vector<float> *featureVector) const
{
    if (!featureVector || image.empty())
    {
        return -1;
    }
// The actual CNN inference logic is implemented in the OrtResNet18Runner class below
#if defined(ENABLE_ONNXRUNTIME)
    class OrtResNet18Runner
    {
    public:
        OrtResNet18Runner()
            : env_(ORT_LOGGING_LEVEL_WARNING, "rtor_cnn")
        {
            const char *modelPathEnv = std::getenv("RTOR_CNN_MODEL");
            const std::string modelPath = (modelPathEnv && std::strlen(modelPathEnv) > 0)
                                              ? std::string(modelPathEnv)
                                              : std::string("./data/resnet18-v2-7.onnx");

            Ort::SessionOptions options;
            options.SetIntraOpNumThreads(1);
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), options);

            Ort::AllocatorWithDefaultOptions allocator;
            auto in = session_->GetInputNameAllocated(0, allocator);
            auto out = session_->GetOutputNameAllocated(0, allocator);
            inputName_ = in.get();
            outputName_ = out.get();
        }

        int infer(const cv::Mat &img, std::vector<float> *outVec) const
        {
            if (!outVec || img.empty())
                return -1;

            // 1) Ensure 3-channel BGR
            cv::Mat bgr;
            if (img.channels() == 1)
                cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
            else if (img.channels() == 3)
                bgr = img;
            else if (img.channels() == 4)
                cv::cvtColor(img, bgr, cv::COLOR_BGRA2BGR);
            else
                return -1;

            // 2) Resize to 224x224
            cv::Mat resized;
            cv::resize(bgr, resized, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);

            // 3) BGR -> RGB
            cv::Mat rgb;
            cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

            // 4) Convert to float in [0,1]
            cv::Mat rgb32f;
            rgb.convertTo(rgb32f, CV_32F, 1.0 / 255.0);

            // 5) Normalize per channel and pack to NCHW
            // ImageNet mean/std for ResNet18
            const float mean[3] = {0.485f, 0.456f, 0.406f};
            const float stdv[3] = {0.229f, 0.224f, 0.225f};

            const int H = 224, W = 224;
            std::vector<float> input(1 * 3 * H * W);

            // rgb32f is HxWx3, float
            for (int y = 0; y < H; ++y)
            {
                const cv::Vec3f *row = rgb32f.ptr<cv::Vec3f>(y);
                for (int x = 0; x < W; ++x)
                {
                    const cv::Vec3f &px = row[x]; // (R,G,B) in [0,1]
                    const float r = (px[0] - mean[0]) / stdv[0];
                    const float g = (px[1] - mean[1]) / stdv[1];
                    const float b = (px[2] - mean[2]) / stdv[2];

                    // NCHW indexing: c*H*W + y*W + x
                    input[0 * H * W + y * W + x] = r;
                    input[1 * H * W + y * W + x] = g;
                    input[2 * H * W + y * W + x] = b;
                }
            }

            // 6) Create tensor and run
            const std::array<int64_t, 4> shape = {1, 3, 224, 224};
            Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
                OrtAllocatorType::OrtArenaAllocator,
                OrtMemType::OrtMemTypeDefault);

            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memInfo,
                input.data(),
                input.size(),
                shape.data(),
                shape.size());

            const char *inputNames[] = {inputName_.c_str()};
            const char *outputNames[] = {outputName_.c_str()};

            auto outputs = session_->Run(
                Ort::RunOptions{nullptr},
                inputNames,
                &inputTensor,
                1,
                outputNames,
                1);

            if (outputs.empty() || !outputs[0].IsTensor())
                return -1;

            float *tensorData = outputs[0].GetTensorMutableData<float>();
            const size_t count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
            outVec->assign(tensorData, tensorData + count);
            return outVec->empty() ? -1 : 0;
        }

    private:
        Ort::Env env_;
        std::unique_ptr<Ort::Session> session_;
        std::string inputName_;
        std::string outputName_;
    };

    try
    {
        static OrtResNet18Runner runner;
        return runner.infer(image, featureVector);
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "[CNN] ONNX Runtime inference failed: %s\n", e.what());
        return -1;
    }
#else
    static bool warned = false;
    if (!warned)
    {
        warned = true;
        std::fprintf(
            stderr,
            "[CNN] ONNX Runtime is disabled. Build with ONNXRUNTIME_DIR=...; default model path is ./data/resnet18-v2-7.onnx (override with RTOR_CNN_MODEL).\n");
    }
    (void)image;
    featureVector->clear();
    return -1;
#endif
}
