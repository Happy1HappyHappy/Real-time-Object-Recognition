/*
Claire Liu, Yu-Jing Wei
IExtractor.hpp

Path: include/IExtractor.hpp
Description: Declares the IExtractor interface for feature extraction.
*/

#pragma once

#include "extractorFactory.hpp"
#include "regionAnalyzer.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/*
IExtractor interface for feature extraction.
public:

protected:

*/
class IExtractor
{
public:
    virtual ~IExtractor() = default;
    virtual int extract(const char *imagePath, std::vector<float> *out) const
    {
        // Load the image from the given path
        cv::Mat img = cv::imread(imagePath);
        if (img.empty())
            return -1;
        return extractMat(img, out);
    }

    virtual int extractMat(const cv::Mat &image, std::vector<float> *out) const = 0;
    virtual int extractRegion(const RegionFeatures &region, std::vector<float> *out) const
    {
        (void)region;
        (void)out;
        return -1;
    }

    virtual std::string type() const { return ExtractorFactory::extractorTypeToString(type_); }

protected:
    ExtractorType type_;
    explicit IExtractor(ExtractorType type) : type_(type) {}
};
