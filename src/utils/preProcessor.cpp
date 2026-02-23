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
#include <unordered_map>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <opencv2/opencv.hpp>

cv::Mat PreProcessor::filterLabelsByMinArea(const cv::Mat &binary, int minAreaPixels)
{
  CV_Assert(!binary.empty());
  CV_Assert(binary.type() == CV_8U);

  cv::Mat stats;
  cv::Mat centroids;
  cv::Mat ccLabels;
  const int numLabels = cv::connectedComponentsWithStats(binary, ccLabels, stats, centroids, 8, CV_32S);
  if (numLabels <= 1)
  {
    return cv::Mat::zeros(binary.size(), CV_32S);
  }

  cv::Mat filteredLabels = cv::Mat::zeros(ccLabels.size(), CV_32S);
  std::unordered_map<int, int> remap;
  int nextId = 1;
  for (int id = 1; id < numLabels; ++id)
  {
    const int area = stats.at<int>(id, cv::CC_STAT_AREA);
    if (area >= minAreaPixels)
    {
      remap[id] = nextId++;
    }
  }

  for (int y = 0; y < ccLabels.rows; ++y)
  {
    const int *srcRow = ccLabels.ptr<int>(y);
    int *dstRow = filteredLabels.ptr<int>(y);
    for (int x = 0; x < ccLabels.cols; ++x)
    {
      const int oldId = srcRow[x];
      if (oldId <= 0)
        continue;
      auto it = remap.find(oldId);
      if (it != remap.end())
      {
        dstRow[x] = it->second;
      }
    }
  }
  return filteredLabels;
}

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
  // Analyze the labeled regions to extract features and find the best candidate
  const int frameArea = input.rows * input.cols;
  const int minAreaPixels = std::max(500, frameArea / 50);

  // Connected components + min-area filtering + sequential relabeling.
  regionLabels = filterLabelsByMinArea(cleanedBinary, minAreaPixels);

  // Colorize the post-filter labels for visualization
  result.regionIdVis = RegionDetect::colorizeRegionLabels(regionLabels);

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
  cv::Mat value = channels[2];

  // 3. Create a mask for high saturation areas
  cv::Mat satMask;
  cv::threshold(saturation, satMask, satThreshold, 255, cv::THRESH_BINARY);
  // 3b. Create a mask for highlight regions (very high V in HSV)
  cv::Mat highlightMask;
  cv::threshold(value, highlightMask, 230, 255, cv::THRESH_BINARY);
  cv::Mat suppressMask;
  cv::bitwise_or(satMask, highlightMask, suppressMask);

  // 4. Darken grayscale values in saturated/highlight regions to suppress bright noise/speculars
  cv::Mat grayFloat;
  gray.convertTo(grayFloat, CV_32F);

  cv::Mat darkened = grayFloat * alpha;
  darkened.copyTo(grayFloat, suppressMask);

  // 5. Convert back to 8-bit
  grayFloat.convertTo(gray, CV_8U);

  return gray;
}
