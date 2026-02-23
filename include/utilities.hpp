#pragma once

#include <opencv2/opencv.hpp>
#include "regionAnalyzer.hpp"

namespace utilities
{
// Rotate ROI by region primary axis and resize for embedding model input.
bool prepEmbeddingImage(
    const cv::Mat &frame,
    const RegionFeatures &region,
    cv::Mat &embImage,
    int outputSize = 224,
    bool debug = false);
}

