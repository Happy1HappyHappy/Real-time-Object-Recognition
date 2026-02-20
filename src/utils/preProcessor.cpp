/*
  Claire Liu, Yu-Jing Wei
  preProcessor.cpp

  Path: src/utils/preProcessor.cpp
  Description: Pre-processes images before feature extraction.
*/

#include "preProcessor.hpp"
#include "regionDetect.hpp"
#include "thresholding.hpp"
#include <opencv2/opencv.hpp>

cv::Mat PreProcessor::process(const cv::Mat &input)
{
    // cv thresholding to get a binary image
    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);

    cv::Mat binary;

    // temp for grassfire output
    cv::Mat output;
    

    Threadsholding::dynamicThreadsHold(gray, binary);

    // Region detection using grassfire algorithm
    RegionDetect::grassfire(binary, output);

    return output;
}