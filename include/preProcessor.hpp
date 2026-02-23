/*
  Claire Liu, Yu-Jing Wei
  preProcessor.hpp

  Path: include/preProcessor.hpp
  Description: Header file for preProcessor.cpp to pre-process images before feature extraction.
*/

#pragma once // Include guard

#include <opencv2/opencv.hpp>
#include "regionAnalyzer.hpp"

struct DetectionResult
{
    bool valid = false;
    cv::Mat thresholdedImage;
    cv::Mat cleanedImage;
    std::vector<RegionFeatures> regions;
    std::vector<cv::Rect> regionBBoxes;
    std::vector<cv::Mat> regionEmbImages;
    cv::Mat regionIdVis;
    RegionFeatures bestRegion;
    cv::Rect bestBBox;
    cv::Mat embImage;
    cv::Mat debugFrame;
};

class PreProcessor
{
public:
    static DetectionResult detect(const cv::Mat &input, bool keepAllRegions);
    static DetectionResult detect(const cv::Mat &input);
    static cv::Mat process(const cv::Mat &input, cv::Mat &output);
    static cv::Mat imgPreProcess(
        const cv::Mat &input,
        float alpha = 0.5f,
        int satThreshold = 50,
        int blurKernel = 5);

private:
    static cv::Mat filterLabelsByMinArea(const cv::Mat &binary, int minAreaPixels);
};
