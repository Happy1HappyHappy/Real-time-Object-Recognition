/*
  Claire Liu, Yu-Jing Wei
  preProcessor.cpp

  Path: src/utils/preProcessor.cpp
  Description: Pre-processes images before feature extraction.
*/

#include "preProcessor.hpp"
#include "distanceTransform.hpp"
#include "regionDetect.hpp"
#include "regionAnalyzer.hpp"
#include "thresholding.hpp"
#include "morphologicalFilter.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <opencv2/opencv.hpp>

DetectionResult PreProcessor::detect(const cv::Mat &input)
{
  DetectionResult result;
  CV_Assert(!input.empty());

  MorphologicalFilter myFilter;

  cv::Mat gray;
  cv::Mat binary;
  cv::Mat cleanedBinary;
  cv::Mat regionLabels;

  // Convert to grayscale and apply dynamic thresholding
  cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
  Threadsholding::dynamicThreadsHold(gray, binary);
  myFilter.defaultDilationErosion(binary, cleanedBinary);

  // Connected component labeling to find regions
  RegionDetect::twoPassSegmentation(cleanedBinary, regionLabels);

  // Analyze the labeled regions to extract features and find the best candidate
  const int frameArea = input.rows * input.cols;
  const int minAreaPixels = std::max(2000, frameArea / 300); // ~0.33% of frame
  RegionAnalyzer analyzer(RegionAnalyzer::Params(false, minAreaPixels, true));
  auto regions = analyzer.analyzeLabels(regionLabels);

  result.debugFrame = input.clone();
  result.embImage.release();

  if (regions.empty())
  {
    return result;
  }

  const double kAreaWeight = 0.8;
  const double kDistWeight = 0.2;
  std::sort(regions.begin(), regions.end(),
            [](const RegionFeatures &a, const RegionFeatures &b)
            {
              return a.area > b.area;
            });

  const size_t kScoredCandidates = std::min<size_t>(regions.size(), 5);
  size_t bestIdx = 0;
  double bestScore = -1.0;
  double bestPeakDist = 0.0;

  for (size_t i = 0; i < kScoredCandidates; ++i)
  {
    const auto &r = regions[i];
    cv::Rect box = cv::boundingRect(r.contour);
    box &= cv::Rect(0, 0, input.cols, input.rows);
    if (box.width <= 0 || box.height <= 0)
      continue;

    cv::Mat roiBinary = cleanedBinary(box).clone();
    cv::Mat roiDist;
    DistanceTransform::grassfire(roiBinary, roiDist);

    std::vector<cv::Point> shifted;
    shifted.reserve(r.contour.size());
    for (const auto &p : r.contour)
    {
      shifted.emplace_back(p.x - box.x, p.y - box.y);
    }
    cv::Mat roiMask = cv::Mat::zeros(roiDist.size(), CV_8U);
    std::vector<std::vector<cv::Point>> cs = {shifted};
    cv::drawContours(roiMask, cs, 0, cv::Scalar(255), cv::FILLED);

    double peakDist = 0.0;
    cv::minMaxLoc(roiDist, nullptr, &peakDist, nullptr, nullptr, roiMask);

    const double areaNorm = r.area / static_cast<double>(std::max(frameArea, 1));
    const double distNorm = peakDist / static_cast<double>(std::max(box.width, box.height));
    const double score = kAreaWeight * areaNorm + kDistWeight * distNorm;
    if (score > bestScore)
    {
      bestScore = score;
      bestIdx = i;
      bestPeakDist = peakDist;
    }
  }
  const RegionFeatures &best = regions[bestIdx];

  const size_t maxDebugBoxes = std::min<size_t>(regions.size(), 3);
  for (size_t i = 0; i < maxDebugBoxes; ++i)
  {
    const auto &r = regions[i];
    cv::Rect box = cv::boundingRect(r.contour);
    box &= cv::Rect(0, 0, input.cols, input.rows);
    if (box.width <= 0 || box.height <= 0)
      continue;
    cv::rectangle(result.debugFrame, box, cv::Scalar(255, 180, 0), 1, cv::LINE_AA);
  }

  cv::Rect bbox = cv::boundingRect(best.contour);
  bbox &= cv::Rect(0, 0, input.cols, input.rows);
  if (bbox.width > 0 && bbox.height > 0)
  {
    result.embImage = input(bbox).clone();
  }

  cv::rectangle(result.debugFrame, bbox, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
  cv::circle(result.debugFrame, best.centroid, 3, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
  cv::putText(result.debugFrame,
              "best area=" + std::to_string(static_cast<int>(best.area)) +
                  " peak=" + std::to_string(static_cast<int>(bestPeakDist)) +
                  " regions=" + std::to_string(regions.size()),
              cv::Point(bbox.x, std::max(20, bbox.y - 10)),
              cv::FONT_HERSHEY_SIMPLEX,
              0.5,
              cv::Scalar(0, 255, 0),
              1,
              cv::LINE_AA);

  result.valid = !result.embImage.empty();
  result.bestRegion = best;
  result.bestBBox = bbox;
  return result;
}

cv::Mat PreProcessor::process(const cv::Mat &input, cv::Mat &output)
{
  DetectionResult result = detect(input);
  output = result.embImage;
  return result.debugFrame;
}
