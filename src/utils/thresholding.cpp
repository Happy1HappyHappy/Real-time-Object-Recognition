/*
  Claire Liu, Yu-Jing Wei
  thresholding.cpp
  Path: src/utils/thresholding.cpp
  Description: Provides dynamic thresholding functionality using K-means clustering.
*/

#include "thresholding.hpp"
#include <opencv2/opencv.hpp>

/*
dynamicThreshold applies K-means clustering to determine an optimal threshold for binarizing the input image.
It converts the input image to grayscale if necessary, reshapes it for K-means, and then performs clustering
to find two centers. The threshold is set to the midpoint between these centers, and the output binary image
is created using this threshold. The function also inverts the binary image to ensure that darker objects become
foreground, which is suitable for white-background scenes.
*/
void Thresholding::dynamicThreshold(const cv::Mat &src, cv::Mat &dst)
{
    // Convert image to grey scale if it's not already
    cv::Mat gray;
    if (src.channels() == 3)
    {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = src.clone();
    }

    // Pre process data for k-means calculation
    cv::Mat data;
    gray.convertTo(data, CV_32F);
    data = data.reshape(1, (int)data.total());

    // K-means clustering
    int K = 2;
    cv::Mat labels, centers;
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0);

    // attemp 3 times to get the best result
    cv::kmeans(data, K, labels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

    float c1 = centers.at<float>(0, 0);
    float c2 = centers.at<float>(1, 0);

    // take the middle of the two centers as the threshold value
    int thresholdValue = static_cast<int>((c1 + c2) / 2.0);

    // For white-background scenes, invert so darker objects become foreground.
    cv::threshold(gray, dst, thresholdValue, 255, cv::THRESH_BINARY_INV);
}
