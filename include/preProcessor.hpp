/*
  Claire Liu, Yu-Jing Wei
  preProcessor.hpp

  Path: include/preProcessor.hpp
  Description: Header file for preProcessor.cpp to pre-process images before feature extraction.
*/

#pragma once // Include guard

#include <opencv2/opencv.hpp>

class PreProcessor
{
public:
    static cv::Mat process(const cv::Mat &input, cv::Mat &output);
};