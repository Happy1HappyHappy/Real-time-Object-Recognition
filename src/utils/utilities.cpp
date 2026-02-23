/*
  Claire Liu, Yu-Jing Wei
  utilities.cpp
  Path: src/utils/utilities.cpp
  Description: Provides utility functions for image processing and region handling.
*/

#include "utilities.hpp"

#include <array>
#include <opencv2/opencv.hpp>

namespace
{
    // Helper function to apply an affine transformation to a point using a 2x3 matrix.
    static inline cv::Point2f affinePoint(const cv::Mat &m, const cv::Point2f &p)
    {
        const double x = m.at<double>(0, 0) * p.x + m.at<double>(0, 1) * p.y + m.at<double>(0, 2);
        const double y = m.at<double>(1, 0) * p.x + m.at<double>(1, 1) * p.y + m.at<double>(1, 2);
        return cv::Point2f(static_cast<float>(x), static_cast<float>(y));
    }
}

/*
prepEmbeddingImage prepares an embedding image for a given region by performing rotation and cropping.
It takes the input frame and region features, applies an affine transformation to align the region
with the horizontal axis, crops the aligned region, and resizes it to the specified output size.
The resulting embedding image is suitable for input to a CNN extractor. The function returns true if
the preparation was successful.
*/
bool utilities::prepEmbeddingImage(
    const cv::Mat &frame,
    const RegionFeatures &region,
    cv::Mat &embImage,
    int outputSize,
    bool debug)
{
    // Release any existing data in the output embedding image before processing.
    // This ensures that if the preparation fails, the output will be empty rather than containing stale data.
    embImage.release();
    if (frame.empty() || outputSize <= 0 || region.area <= 0.0)
    {
        return false;
    }

    // Compute the rotation matrix to align the region's primary axis with the horizontal axis,
    // using the region's centroid as the center of rotation.
    const cv::Point2f center = region.centroid;
    const double angleDeg = -static_cast<double>(region.theta) * 180.0 / CV_PI;
    cv::Mat rotM = cv::getRotationMatrix2D(center, angleDeg, 1.0);
    // Apply the affine transformation to the input frame to obtain a rotated image where the region is aligned horizontally
    cv::Mat rotated;
    cv::warpAffine(frame, rotated, rotM, frame.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    // Get the corners of the region's oriented bounding box (OBB) and apply the same affine transformation to them
    // to find the corresponding axis-aligned bounding box in the rotated image
    cv::Point2f obbPts[4];
    region.orientedBBox.points(obbPts);
    std::array<cv::Point2f, 4> rpts = {
        affinePoint(rotM, obbPts[0]),
        affinePoint(rotM, obbPts[1]),
        affinePoint(rotM, obbPts[2]),
        affinePoint(rotM, obbPts[3])};
    // Compute the axis-aligned bounding box of the transformed OBB corners to determine the cropping region in the rotated image
    std::vector<cv::Point2f> pts(rpts.begin(), rpts.end());
    cv::Rect roi = cv::boundingRect(pts) & cv::Rect(0, 0, rotated.cols, rotated.rows);
    if (roi.width <= 1 || roi.height <= 1)
    {
        return false;
    }
    // Crop the aligned region from the rotated image and resize it to the desired output size for CNN input
    cv::Mat cropped = rotated(roi).clone();
    cv::resize(cropped, embImage, cv::Size(outputSize, outputSize), 0, 0, cv::INTER_LINEAR);

    if (debug)
    {
        cv::imshow("cnn_prepped", embImage);
    }
    return !embImage.empty();
}
