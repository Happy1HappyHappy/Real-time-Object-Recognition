#include "utilities.hpp"

#include <array>
#include <opencv2/opencv.hpp>

namespace
{
static inline cv::Point2f affinePoint(const cv::Mat &m, const cv::Point2f &p)
{
    const double x = m.at<double>(0, 0) * p.x + m.at<double>(0, 1) * p.y + m.at<double>(0, 2);
    const double y = m.at<double>(1, 0) * p.x + m.at<double>(1, 1) * p.y + m.at<double>(1, 2);
    return cv::Point2f(static_cast<float>(x), static_cast<float>(y));
}
}

bool utilities::prepEmbeddingImage(
    const cv::Mat &frame,
    const RegionFeatures &region,
    cv::Mat &embImage,
    int outputSize,
    bool debug)
{
    embImage.release();
    if (frame.empty() || outputSize <= 0 || region.area <= 0.0)
    {
        return false;
    }

    const cv::Point2f center = region.centroid;
    const double angleDeg = -static_cast<double>(region.theta) * 180.0 / CV_PI;
    cv::Mat rotM = cv::getRotationMatrix2D(center, angleDeg, 1.0);

    cv::Mat rotated;
    cv::warpAffine(frame, rotated, rotM, frame.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    cv::Point2f obbPts[4];
    region.orientedBBox.points(obbPts);
    std::array<cv::Point2f, 4> rpts = {
        affinePoint(rotM, obbPts[0]),
        affinePoint(rotM, obbPts[1]),
        affinePoint(rotM, obbPts[2]),
        affinePoint(rotM, obbPts[3])};

    std::vector<cv::Point2f> pts(rpts.begin(), rpts.end());
    cv::Rect roi = cv::boundingRect(pts) & cv::Rect(0, 0, rotated.cols, rotated.rows);
    if (roi.width <= 1 || roi.height <= 1)
    {
        return false;
    }

    cv::Mat cropped = rotated(roi).clone();
    cv::resize(cropped, embImage, cv::Size(outputSize, outputSize), 0, 0, cv::INTER_LINEAR);

    if (debug)
    {
        cv::imshow("cnn_prepped", embImage);
    }
    return !embImage.empty();
}

