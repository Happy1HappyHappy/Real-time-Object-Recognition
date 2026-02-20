/*
Claire Liu, Yu-Jing Wei
Extractor.hpp

Path: include/extractor.hpp
Description: Header file for extractor.cpp to extract features from images.
*/

#pragma once

#include "IExtractor.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/core.hpp>

struct BaselineExtractor : public IExtractor
{
    BaselineExtractor(ExtractorType type) : IExtractor(type) {}
    // Override the extractMat function to implement the feature extraction logic for the baseline extractor
    int extractMat(const cv::Mat &image, std::vector<float> *featureVector) const override;
};

struct CNNExtractor : public IExtractor
{
    CNNExtractor(ExtractorType type) : IExtractor(type) {}
    // Override the extractMat function to implement the feature extraction logic for the ResNet extractor
    int extractMat(const cv::Mat &image, std::vector<float> *featureVector) const override;
};

struct EigenspaceExtractor : public IExtractor
{
    EigenspaceExtractor(ExtractorType type) : IExtractor(type) {}
    // Override the extractMat function to implement the feature extraction logic for the eigenspace extractor
    int extractMat(const cv::Mat &image, std::vector<float> *featureVector) const override;
};
