/*
Claire Liu, Yu-Jing Wei
regionDetect.hpp

Path: include/regionDetect.hpp
Description: Region detection utilities (grassfire / connected components).
*/

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

class RegionDetect
{
public:
    RegionDetect();
    ~RegionDetect();
    static void grassfire(cv::Mat &binaryImage, cv::Mat &regionMap);
    static void twoSegmentation(cv::Mat &binaryImage, cv::Mat &regionMap);

private:
};
