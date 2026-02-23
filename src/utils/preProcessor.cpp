/*
  Claire Liu, Yu-Jing Wei
  preProcessor.cpp

  Path: src/utils/preProcessor.cpp
  Description: Pre-processes images before feature extraction.
*/

#include "preProcessor.hpp"
#include "regionDetect.hpp"
#include "regionAnalyzer.hpp"
#include "thresholding.hpp"
#include "morphologicalFilter.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <opencv2/opencv.hpp>

DetectionResult PreProcessor::detect(const cv::Mat &input, bool keepAllRegions)
{
  DetectionResult result;
  CV_Assert(!input.empty());

  cv::Mat gray;
  cv::Mat binary;
  cv::Mat cleanedBinary;
  cv::Mat regionLabels;

  // Pre-process the image to enhance features and suppress noise
  gray = imgPreProcess(input, 0.5f, 50, 5);
  // Dynamic thresholding to get binary image
  Threadsholding::dynamicThreadsHold(gray, binary);
  result.thresholdedImage = binary.clone();
  // Morphological operations to clean up the binary image
  MorphologicalFilter myFilter;
  myFilter.defaultDilationErosion(binary, cleanedBinary);
  result.cleanedImage = cleanedBinary.clone();
  // Connected component labeling to find regions
  RegionDetect::twoPassSegmentation(cleanedBinary, regionLabels);
  // Colorize the region labels for visualization
  result.regionIdVis = RegionDetect::colorizeRegionLabels(regionLabels);

  // Analyze the labeled regions to extract features and find the best candidate
  const int frameArea = input.rows * input.cols;
  const int minAreaPixels = std::max(500, frameArea / 50);
  RegionAnalyzer analyzer(RegionAnalyzer::Params(
      /*keepMasks*/ false,
      minAreaPixels,
      /*externalOnly*/ true));
  auto regions = analyzer.analyzeLabels(regionLabels);

  // Initialize the DetectionResult with default values and debug visualization
  result.debugFrame = input.clone();
  result.embImage.release();
  result.regions.clear();
  result.regionBBoxes.clear();
  result.regionEmbImages.clear();

  // If no valid regions are found, return the result with valid=false and empty fields.
  if (regions.empty())
  {
    return result;
  }
  if (keepAllRegions)
  {
    result.regions = regions;
  }

  std::sort(regions.begin(), regions.end(),
            [](const RegionFeatures &a, const RegionFeatures &b)
            {
              return a.area > b.area;
            });

  // Keep all regions > minArea for downstream classification.
  // bestRegion is the largest-area region.
  const RegionFeatures &best = regions.front();
  if (keepAllRegions)
  {
    result.regionBBoxes.reserve(regions.size());
    result.regionEmbImages.reserve(regions.size());
    for (const auto &r : regions)
    {
      cv::Rect box = r.orientedBBox.boundingRect();
      box &= cv::Rect(0, 0, input.cols, input.rows);
      if (box.width <= 0 || box.height <= 0)
        continue;
      result.regionBBoxes.push_back(box);
      result.regionEmbImages.push_back(input(box).clone());
    }
  }
  // Extract the embedding image for the best region (largest area)
  // to be used for classification.
  cv::Rect bbox = best.orientedBBox.boundingRect();
  bbox &= cv::Rect(0, 0, input.cols, input.rows);
  if (bbox.width > 0 && bbox.height > 0)
  {
    result.embImage = input(bbox).clone();
  }
  result.valid = !result.embImage.empty();
  result.bestRegion = best;
  result.bestBBox = bbox;
  return result;
}

DetectionResult PreProcessor::detect(const cv::Mat &input)
{
  return detect(input, true);
}

cv::Mat PreProcessor::process(const cv::Mat &input, cv::Mat &output)
{
  DetectionResult result = detect(input, false);
  output = result.embImage;
  return result.debugFrame;
}

cv::Mat PreProcessor::imgPreProcess(
    const cv::Mat &input,
    float alpha,
    int satThreshold,
    int blurKernel)
{
  CV_Assert(!input.empty());
  CV_Assert(input.type() == CV_8UC3);

  // 1. Grayscale conversion and Gaussian blur to reduce noise
  cv::Mat gray;
  cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, gray, cv::Size(blurKernel, blurKernel), 0);

  // 2. Convert to HSV to analyze saturation
  cv::Mat hsv;
  cv::cvtColor(input, hsv, cv::COLOR_BGR2HSV);

  std::vector<cv::Mat> channels;
  cv::split(hsv, channels);
  cv::Mat saturation = channels[1];

  // 3. Create a mask for high saturation areas
  cv::Mat satMask;
  cv::threshold(saturation, satMask, satThreshold, 255, cv::THRESH_BINARY);

  // 4. Darken the grayscale values in high saturation areas to suppress bright noise
  cv::Mat grayFloat;
  gray.convertTo(grayFloat, CV_32F);

  cv::Mat darkened = grayFloat * alpha;
  darkened.copyTo(grayFloat, satMask);

  // 5. Convert back to 8-bit
  grayFloat.convertTo(gray, CV_8U);

  return gray;
}
