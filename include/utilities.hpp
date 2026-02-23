/*
  Claire Liu, Yu-Jing Wei
  utilities.hpp

  Path: include/utilities.hpp
  Description: Header file for utility functions used in the project.
*/

#pragma once

#include <opencv2/opencv.hpp>
#include "regionAnalyzer.hpp"

// This namespace contains utility functions for image processing and region analysis.
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
