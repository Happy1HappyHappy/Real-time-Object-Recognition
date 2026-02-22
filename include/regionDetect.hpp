/*
  Claire Liu, Yu-Jing Wei
  regionDetect.hpp

  Path: include/regionDetect.hpp
  Description: Header file for regionDetect.cpp to apply morphological filters to images.
*/

#pragma once // Include guard
#include <opencv2/opencv.hpp>

class RegionDetect{
public:
  void twoPassSegmentation(cv::Mat &src, cv::Mat &dst);
    
private:

};