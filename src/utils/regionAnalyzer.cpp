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

float RegionAnalyzer::primaryAxisTheta(double mu20, double mu02, double mu11)
{
    // Primary axis angle (radians):
    // theta = 0.5 * atan2(2*mu11, mu20 - mu02)
    return 0.5f * (float)std::atan2(2.0 * mu11, (mu20 - mu02));
}

cv::Mat RegionAnalyzer::contourToMask(const cv::Size &sz, const std::vector<cv::Point> &contour)
{
    cv::Mat mask(sz, CV_8U, cv::Scalar(0));
    std::vector<std::vector<cv::Point>> cs = {contour};
    cv::drawContours(mask, cs, 0, cv::Scalar(255), cv::FILLED);
    return mask;
}

void RegionAnalyzer::computePixelCentralMoments(
    const cv::Mat &regionMask,
    const cv::Rect &roi,
    const cv::Point2f &c,
    double &mu20, double &mu02, double &mu11)
{
    // Pixel-based central moments over the filled region:
    // mu20 = sum (x-cx)^2, mu02 = sum (y-cy)^2, mu11 = sum (x-cx)(y-cy)
    mu20 = mu02 = mu11 = 0.0;

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

void RegionAnalyzer::computeAxisExtentsFromContour(
    const std::vector<cv::Point> &contour,
    const cv::Point2f &c,
    const cv::Point2f &e1,
    const cv::Point2f &e2,
    float &minE1, float &maxE1,
    float &minE2, float &maxE2)
{
    // Project contour points into the (e1,e2) coordinate system centered at centroid.
    // For each point p: v = p - c
    //   u1 = dot(v, e1)
    //   u2 = dot(v, e2)
    // Then min/max of u1 and u2 become extents along primary/secondary axes.
    float mn1 = std::numeric_limits<float>::infinity();
    float mx1 = -std::numeric_limits<float>::infinity();
    float mn2 = std::numeric_limits<float>::infinity();
    float mx2 = -std::numeric_limits<float>::infinity();

    for (const auto &p : contour)
    {
        cv::Point2f v((float)p.x - c.x, (float)p.y - c.y);
        float u1 = v.x * e1.x + v.y * e1.y;
        float u2 = v.x * e2.x + v.y * e2.y;

        mn1 = std::min(mn1, u1);
        mx1 = std::max(mx1, u1);
        mn2 = std::min(mn2, u2);
        mx2 = std::max(mx2, u2);
    }

    // If contour is empty, keep zeros (should not happen if valid contour).
    if (!std::isfinite(mn1) || !std::isfinite(mx1) || !std::isfinite(mn2) || !std::isfinite(mx2))
    {
        minE1 = maxE1 = minE2 = maxE2 = 0.f;
        return;
    }

    minE1 = mn1;
    maxE1 = mx1;
    minE2 = mn2;
    maxE2 = mx2;
}

std::vector<RegionFeatures> RegionAnalyzer::analyzeLabels(const cv::Mat &labels_32s) const
{
    CV_Assert(!labels_32s.empty());
    CV_Assert(labels_32s.type() == CV_32S);

    double minLabel = 0.0;
    double maxLabel = 0.0;
    cv::minMaxLoc(labels_32s, &minLabel, &maxLabel);

    std::vector<RegionFeatures> regions;
    if (maxLabel < 1.0)
    {
        return regions;
    }
    regions.reserve(static_cast<size_t>(maxLabel));

    for (int label = 1; label <= static_cast<int>(maxLabel); ++label)
    {
        cv::Mat componentMask;
        cv::compare(labels_32s, label, componentMask, cv::CMP_EQ);
        if (cv::countNonZero(componentMask) < params_.minAreaPixels)
            continue;

        std::vector<std::vector<cv::Point>> contours;
        int mode = params_.externalOnly ? cv::RETR_EXTERNAL : cv::RETR_TREE;
        cv::findContours(componentMask.clone(), contours, mode, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty())
            continue;

        auto bestContourIt = std::max_element(
            contours.begin(), contours.end(),
            [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b)
            {
                return std::abs(cv::contourArea(a)) < std::abs(cv::contourArea(b));
            });
        if (bestContourIt == contours.end() || bestContourIt->size() < 3)
            continue;

        RegionFeatures r;
        r.id = label;
        r.contour = *bestContourIt;

        r.area = std::abs(cv::contourArea(r.contour));
        if ((int)r.area < params_.minAreaPixels)
            continue;

        cv::Moments m = cv::moments(r.contour, /*binaryImage=*/false);
        if (std::abs(m.m00) < 1e-9)
            continue;
        r.centroid = cv::Point2f((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
        r.orientedBBox = cv::minAreaRect(r.contour);

        cv::Mat regionMask = contourToMask(labels_32s.size(), r.contour);
        cv::Rect roi = cv::boundingRect(r.contour);

        computePixelCentralMoments(regionMask, roi, r.centroid, r.mu20, r.mu02, r.mu11);
        r.theta = primaryAxisTheta(r.mu20, r.mu02, r.mu11);
        r.e1 = cv::Point2f(std::cos(r.theta), std::sin(r.theta));
        r.e2 = cv::Point2f(-r.e1.y, r.e1.x);
        computeAxisExtentsFromContour(r.contour, r.centroid, r.e1, r.e2,
                                      r.minE1, r.maxE1, r.minE2, r.maxE2);

        // Optionally keep the binary mask for the region
        // (not needed for this project, but could be useful for debugging or future extensions).
        if (params_.keepMasks)
        {
            r.mask = regionMask;
        }

        const double obbArea = r.orientedBBox.size.width * r.orientedBBox.size.height;
        r.percentFilled = (obbArea > 1e-6) ? (r.area / obbArea) : 0.0;

        const double w = r.orientedBBox.size.width;
        const double h = r.orientedBBox.size.height;
        r.aspectRatio = (h > 1e-6) ? ((w > h) ? (w / h) : (h / w)) : 0.0;

        cv::Moments mm = cv::moments(regionMask, true);
        cv::HuMoments(mm, r.hu);
        for (int i = 0; i < 7; ++i)
        {
            if (r.hu[i] != 0.0)
            {
                r.hu[i] = -1.0 * std::copysign(1.0, r.hu[i]) * std::log10(std::abs(r.hu[i]));
            }
        }

        regions.push_back(std::move(r));
    }

    return regions;
}

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
