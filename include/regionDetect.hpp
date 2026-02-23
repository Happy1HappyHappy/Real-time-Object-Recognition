/*
  Claire Liu, Yu-Jing Wei
  regionDetect.hpp

  Path: include/regionDetect.hpp
  Description: Header file for regionDetect.cpp to detect and analyze regions in images.
*/

#pragma once

#include <opencv2/opencv.hpp>

/*
RegionDetect class provides methods for segmenting binary images into connected regions
using a two-pass algorithm. It also includes a utility function to visualize the segmented
regions by colorizing the label map with random colors.
*/
class RegionDetect
{
public:
  // Input: binary CV_8U image (0 background, non-zero foreground)
  // Output: CV_32S label image (0 background, 1..N regions)
  static void twoPassSegmentation(const cv::Mat &binaryImage, cv::Mat &regionMap);

  // Visualization-only utility: colorize CV_32S label map with random colors.
  static cv::Mat colorizeRegionLabels(const cv::Mat &regionMap32S, uint64_t seed = 0);
};
