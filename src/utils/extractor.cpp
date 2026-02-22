/*
  Claire Liu, Yu-Jing Wei
  extractor.cpp

  Path: src/utils/extractor.cpp
  Description: Extracts features from images.
*/

#include "extractor.hpp"
#include "filters.hpp"
#include "preProcessor.hpp"
#include "regionAnalyzer.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>

int BaselineExtractor::extractMat(
    const cv::Mat &image,
    std::vector<float> *featureVector) const
{
    if (!featureVector || image.empty())
    {
        return -1;
    }

    DetectionResult det = PreProcessor::detect(image);
    if (!det.valid)
    {
        return -1;
    }

    const std::vector<double> shape = getShapeFeatureVector(det.bestRegion);
    featureVector->clear();
    featureVector->reserve(shape.size());
    for (double v : shape)
    {
        featureVector->push_back(static_cast<float>(v));
    }

    return featureVector->empty() ? -1 : 0;
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
