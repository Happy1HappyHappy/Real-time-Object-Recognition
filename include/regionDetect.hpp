/*
  Region detection utilities (connected components).
*/

#pragma once

#include <opencv2/opencv.hpp>

class RegionDetect
{
public:
    // Input: binary CV_8U image (0 background, non-zero foreground)
    // Output: CV_32S label image (0 background, 1..N regions)
    static void twoPassSegmentation(const cv::Mat &binaryImage, cv::Mat &regionMap);
};
