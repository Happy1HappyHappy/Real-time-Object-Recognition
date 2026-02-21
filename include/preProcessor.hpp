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
    RegionFeatures bestRegion;
    cv::Rect bestBBox;
    cv::Mat embImage;
    cv::Mat debugFrame;
};

class PreProcessor
{
public:
    static DetectionResult detect(const cv::Mat &input);
    static cv::Mat process(const cv::Mat &input, cv::Mat &output);
};
