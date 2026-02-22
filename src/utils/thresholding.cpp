#include "thresholding.hpp"
#include <opencv2/opencv.hpp>

// Utlizing K-means to dinamiclly threadshold the image
// The image will be coverted to grey and apply Gassian Blur to reduce noise
// param: CV_8UC3 color image
void Threadsholding::dynamicThreadsHold(const cv::Mat &src, cv::Mat &dst)
{
    // Convert image to grey scale
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
