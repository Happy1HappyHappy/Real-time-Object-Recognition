#include "regionDetect.hpp"

void RegionDetect::twoPassSegmentation(const cv::Mat &binaryImage, cv::Mat &regionMap)
{
    CV_Assert(!binaryImage.empty());
    CV_Assert(binaryImage.type() == CV_8U);

    cv::Mat bin = binaryImage.clone();
    if (bin.channels() != 1)
    {
        cv::cvtColor(bin, bin, cv::COLOR_BGR2GRAY);
    }
    cv::threshold(bin, bin, 0, 255, cv::THRESH_BINARY);

    cv::connectedComponents(bin, regionMap, 8, CV_32S);
}
