/*
  Claire Liu, Yu-Jing Wei
  thresholding.hpp

  Path: include/thresholding.hpp
  Description: Header file for thresholding.cpp to perform dynamic thresholding on images.
*/

#pragma once
#include <opencv2/opencv.hpp>

/*
This class provides a static method for performing dynamic thresholding on images.
The dynamicThreshold method takes a source image (src) and an output image (dst) as parameters.
It applies a dynamic thresholding technique to the source image and stores the result in the
destination image.
*/
class Thresholding
{
public:
    static void dynamicThreshold(const cv::Mat &src, cv::Mat &dst);
};