#pragma once
#include <opencv2/opencv.hpp>

class Threadsholding{
public:
    static void dynamicThreadsHold(const cv::Mat &src, cv::Mat &dst);
};