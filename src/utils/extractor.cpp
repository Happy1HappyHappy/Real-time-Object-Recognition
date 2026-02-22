/*
  Claire Liu, Yu-Jing Wei
  extractor.cpp

  Path: src/utils/extractor.cpp
  Description: Extracts features from images.
*/

#include "extractor.hpp"
#include "filters.hpp"
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
#include <vector>
#include <opencv2/opencv.hpp>

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
    const int minAreaPixels = std::max(2000, frameArea / 10);
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
    // TODO:
    // Pre-process the frame and get the region of interest (ROI)

    // Bounding box of the ROI on the current frame

    // Extract feature vector from the ROI using the CNN extractor

    // For demonstration, we will just return a dummy feature vector of size 512 with all values set to 0.5
    featureVector->assign(512, 0.5f);
    return 0; // Success
}

int EigenspaceExtractor::extractMat(
    const cv::Mat &image,
    std::vector<float> *featureVector) const
{
    // TODO:
    // Pre-process the frame and get the region of interest (ROI)

    // Bounding box of the ROI on the current frame

    // Extract feature vector from the ROI using the eigenspace extractor

    // For demonstration, we will just return a dummy feature vector of size 100 with all values set to 0.5
    featureVector->assign(100, 0.5f);
    return 0; // Success
}
