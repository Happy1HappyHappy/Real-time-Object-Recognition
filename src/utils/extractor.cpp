/*
  Claire Liu, Yu-Jing Wei
  extractor.cpp

  Path: project2/src/utils/extractor.cpp
  Description: Extracts features from images.
*/

#include "extractor.hpp"
#include "filters.hpp"
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
    featureVector->assign(100, 0.5f);
    return 0; // Success
}

int CNNExtractor::extractMat(
    const cv::Mat &image,
    std::vector<float> *featureVector) const
{
    // Placeholder implementation for CNN feature extraction
    // In a real implementation, this would involve loading a pre-trained CNN model
    // and running the image through the model to get the feature vector.

    // For demonstration, we will just return a dummy feature vector of size 512 with all values set to 0.5
    featureVector->assign(512, 0.5f);
    return 0; // Success
}

int EigenspaceExtractor::extractMat(
    const cv::Mat &image,
    std::vector<float> *featureVector) const
{
    // Placeholder implementation for eigenspace feature extraction
    // In a real implementation, this would involve projecting the image onto a pre-computed eigenspace
    // and returning the resulting feature vector.

    // For demonstration, we will just return a dummy feature vector of size 100 with all values set to 0.5
    featureVector->assign(100, 0.5f);
    return 0; // Success
}