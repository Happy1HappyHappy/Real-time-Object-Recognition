/*
  Claire Liu, Yu-Jing Wei
  RegionAnalyzer.cpp
  Path: src/utils/RegionAnalyzer.cpp
  Description: Analyzes connected regions in binary images to extract geometric and second-moment features.
*/

#include "regionAnalyzer.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

/*
primaryAxisTheta computes the angle of the primary axis of a region based on its central moments.
The angle is calculated using the formula:
theta = 0.5 * atan2(2*mu11, mu20 - mu02)
where mu20, mu02, and mu11 are the central moments of the region.
*/
float RegionAnalyzer::primaryAxisTheta(double mu20, double mu02, double mu11)
{
    // Primary axis angle (radians):
    // theta = 0.5 * atan2(2*mu11, mu20 - mu02)
    return 0.5f * (float)std::atan2(2.0 * mu11, (mu20 - mu02));
}

/*
computePixelCentralMoments calculates the central moments (mu20, mu02, mu11) for a given
region defined by a binary mask. It iterates over the pixels in the specified ROI and
accumulates the moments based on the pixel coordinates relative to the centroid.
*/
void RegionAnalyzer::computePixelCentralMoments(
    const cv::Mat &regionMask,
    const cv::Rect &roi,
    const cv::Point2f &c,
    double &mu20, double &mu02, double &mu11)
{
    // Pixel-based central moments over the filled region:
    // mu20 = sum (x-cx)^2, mu02 = sum (y-cy)^2, mu11 = sum (x-cx)(y-cy)
    mu20 = mu02 = mu11 = 0.0;
    // Iterate over the pixels in the ROI and accumulate moments for pixels that
    // belong to the region (non-zero in mask)
    for (int y = roi.y; y < roi.y + roi.height; ++y)
    {
        const uchar *row = regionMask.ptr<uchar>(y);
        for (int x = roi.x; x < roi.x + roi.width; ++x)
        {
            if (row[x] == 0)
                continue;

            const double dx = (double)x - (double)c.x;
            const double dy = (double)y - (double)c.y;
            mu20 += dx * dx;
            mu02 += dy * dy;
            mu11 += dx * dy;
        }
    }
}

/*
computeAxisExtentsFromMask calculates the extents of a region along its primary (e1) and secondary (e2) axes
by projecting the pixels of the region (defined by the binary mask) onto these axes. It iterates over the
pixels in the specified ROI, checks if they belong to the region, and updates the minimum and maximum
projections (minE1, maxE1, minE2, maxE2) accordingly.
*/
void RegionAnalyzer::computeAxisExtentsFromMask(
    const cv::Mat &regionMask,
    const cv::Rect &roi,
    const cv::Point2f &c,
    const cv::Point2f &e1,
    const cv::Point2f &e2,
    float &minE1, float &maxE1,
    float &minE2, float &maxE2)
{
    // Project all region pixels (not just boundary) into the (e1,e2) coordinates.
    float mn1 = std::numeric_limits<float>::infinity();
    float mx1 = -std::numeric_limits<float>::infinity();
    float mn2 = std::numeric_limits<float>::infinity();
    float mx2 = -std::numeric_limits<float>::infinity();
    // Iterate over the pixels in the ROI and update the min/max projections for pixels that belong to the region (non-zero in mask)
    for (int y = roi.y; y < roi.y + roi.height; ++y)
    {
        const uchar *row = regionMask.ptr<uchar>(y);
        for (int x = roi.x; x < roi.x + roi.width; ++x)
        {
            if (row[x] == 0)
                continue;
            cv::Point2f v((float)x - c.x, (float)y - c.y);
            float u1 = v.x * e1.x + v.y * e1.y;
            float u2 = v.x * e2.x + v.y * e2.y;
            // Update min/max projections for the primary axis (e1) and secondary axis (e2)
            mn1 = std::min(mn1, u1);
            mx1 = std::max(mx1, u1);
            mn2 = std::min(mn2, u2);
            mx2 = std::max(mx2, u2);
        }
    }

    // If contour is empty, keep zeros (should not happen if valid contour).
    if (!std::isfinite(mn1) || !std::isfinite(mx1) || !std::isfinite(mn2) || !std::isfinite(mx2))
    {
        minE1 = maxE1 = minE2 = maxE2 = 0.f;
        return;
    }
    // Set the output extents for the primary axis (e1) and secondary axis (e2) based on the computed min/max projections
    minE1 = mn1;
    maxE1 = mx1;
    minE2 = mn2;
    maxE2 = mx2;
}

/*
computeFeaturesForRegion computes various geometric and second-moment features for a given region defined
by its label ID in the labels_32s matrix. It creates a binary mask for the region, calculates moments,
oriented bounding box, percent filled, aspect ratio, and Hu invariant moments. The computed features
are stored in the output RegionFeatures structure.
*/
bool RegionAnalyzer::computeFeaturesForRegion(
    const cv::Mat &labels_32s,
    int regionId,
    RegionFeatures &out) const
{
    CV_Assert(!labels_32s.empty());
    CV_Assert(labels_32s.type() == CV_32S);
    if (regionId <= 0)
        return false;

    // Create a binary mask for the region
    cv::Mat regionMask;
    cv::compare(labels_32s, regionId, regionMask, cv::CMP_EQ); // CV_8U mask
    const int pixels = cv::countNonZero(regionMask);
    if (pixels < params_.minAreaPixels)
        return false;
    // Compute moments for the region
    cv::Moments m = cv::moments(regionMask, true);
    if (std::abs(m.m00) < 1e-9)
        return false;
    // Compute features for the region
    RegionFeatures r;
    r.id = regionId;
    r.area = static_cast<double>(pixels);
    r.centroid = cv::Point2f(static_cast<float>(m.m10 / m.m00),
                             static_cast<float>(m.m01 / m.m00));
    // Compute the oriented bounding box
    std::vector<cv::Point> nz;
    cv::findNonZero(regionMask, nz);
    if (nz.empty())
        return false;
    cv::Rect roi = cv::boundingRect(nz);
    // Compute central moments, rotation theta and primary axis
    computePixelCentralMoments(regionMask, roi, r.centroid, r.mu20, r.mu02, r.mu11);
    r.theta = primaryAxisTheta(r.mu20, r.mu02, r.mu11);
    r.e1 = cv::Point2f(std::cos(r.theta), std::sin(r.theta));
    r.e2 = cv::Point2f(-r.e1.y, r.e1.x);
    // Compute axis extents by projecting all region pixels into the (e1,e2) coordinates
    computeAxisExtentsFromMask(regionMask, roi, r.centroid, r.e1, r.e2,
                               r.minE1, r.maxE1, r.minE2, r.maxE2);
    // Construct the oriented bounding box (OBB) using the centroid, primary axis, and extents
    const float w = std::max(1.0f, r.maxE1 - r.minE1);
    const float h = std::max(1.0f, r.maxE2 - r.minE2);
    const cv::Point2f obbCenter =
        r.centroid + r.e1 * (0.5f * (r.minE1 + r.maxE1)) + r.e2 * (0.5f * (r.minE2 + r.maxE2));
    r.orientedBBox = cv::RotatedRect(obbCenter, cv::Size2f(w, h), r.theta * 180.0f / (float)CV_PI);
    // Store the contour for visualization (not necessarily needed for feature vector)
    if (params_.keepMasks)
    {
        r.mask = regionMask;
    }
    // Compute shape feature vector (percent filled, aspect ratio, Hu moments)
    const double obbArea = static_cast<double>(w) * static_cast<double>(h);
    r.percentFilled = (obbArea > 1e-6) ? (r.area / obbArea) : 0.0;
    r.aspectRatio = (h > 1e-6f) ? ((w > h) ? (w / h) : (h / w)) : 0.0;
    // Hu invariant moments (7 values)
    cv::Moments mm = cv::moments(regionMask, true);
    cv::HuMoments(mm, r.hu);
    for (int i = 0; i < 7; ++i)
    {
        if (r.hu[i] != 0.0)
        {
            r.hu[i] = -1.0 * std::copysign(1.0, r.hu[i]) * std::log10(std::abs(r.hu[i]));
        }
    }
    // Set the output region features structure with the computed features for this region
    out = std::move(r);
    return true;
}

/*
analyzeLabels processes the input labels_32s matrix, which contains integer labels for connected regions,
and extracts features for each region using computeFeaturesForRegion. It iterates over the unique region
IDs in the labels matrix, computes features for each valid region, and returns a vector of RegionFeatures
structures containing the extracted features for all regions.
*/
std::vector<RegionFeatures> RegionAnalyzer::analyzeLabels(const cv::Mat &labels_32s) const
{
    CV_Assert(!labels_32s.empty());
    CV_Assert(labels_32s.type() == CV_32S);
    // Find the unique region IDs in the labels matrix to determine how many regions to analyze
    double minLabel = 0.0;
    double maxLabel = 0.0;
    cv::minMaxLoc(labels_32s, &minLabel, &maxLabel);

    std::vector<RegionFeatures> regions;
    if (maxLabel < 1.0)
    {
        return regions;
    }
    regions.reserve(static_cast<size_t>(maxLabel));
    // Iterate over each region ID and compute features for valid regions, storing the results in the output vector
    for (int label = 1; label <= static_cast<int>(maxLabel); ++label)
    {
        RegionFeatures r;
        // Compute features for the region with the current label ID and add it to the output vector if valid
        if (computeFeaturesForRegion(labels_32s, label, r))
        {
            regions.push_back(std::move(r));
        }
    }

    return regions;
}

/*
getShapeFeatureVector constructs a feature vector for a given region based on its geometric and second-moment features.
It includes the percent filled, aspect ratio, and the 7 Hu invariant moments, resulting in a 9-dimensional feature vector.
*/
std::vector<double> getShapeFeatureVector(const RegionFeatures &r)
{
    std::vector<double> fv;
    fv.reserve(9);

    fv.push_back(r.percentFilled);
    fv.push_back(r.aspectRatio);
    for (int i = 0; i < 7; ++i)
    {
        fv.push_back(r.hu[i]);
    }
    return fv;
}
