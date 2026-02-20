/*
  Claire Liu, Yu-Jing Wei
  extractor.cpp

  Path: project2/src/utils/extractor.cpp
  Description: Extracts features from images.
*/

#include "extractor.hpp"
#include "filters.hpp"
#include "preProcessor.hpp"
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
    // TODO:
    // Pre-process the frame and get the region of interest (ROI)
    PreProcessor::process(image);
    // Extract feature vector from the ROI using the baseline extractor

    // For demonstration, we will just return a dummy feature vector of size 512 with all values set to 0.5
    featureVector->assign(100, 0.5f);
    return 0; // Success
}

int CNNExtractor::extractMat(
    const cv::Mat &image,
    std::vector<float> *featureVector) const
{
    // TODO:
    // Pre-process the frame and get the region of interest (ROI)
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
    // Extract feature vector from the ROI using the eigenspace extractor

    // For demonstration, we will just return a dummy feature vector of size 100 with all values set to 0.5
    featureVector->assign(100, 0.5f);
    return 0; // Success
}