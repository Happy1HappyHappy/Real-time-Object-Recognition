/*
  Claire Liu, Yu-Jing Wei
  preProcessor.cpp

  Path: src/utils/preProcessor.cpp
  Description: Pre-processes images before feature extraction.
*/

#include "preProcessor.hpp"
#include "regionDetect.hpp"
#include "thresholding.hpp"
#include <opencv2/opencv.hpp>

cv::Mat PreProcessor::process(const cv::Mat &input, cv::Mat &output)
{
  // Thresholding to get a binary image
  cv::Mat gray;
  cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);

  cv::Mat binary;
  cv::threshold(gray, binary,
                128, // threshold value
                255, // max value
                cv::THRESH_BINARY);
  cv::Mat processedImg = input.clone();

  // Morphological operations to clean up the binary image

  // Connected components using grassfire algorithm
  RegionDetect::grassfire(binary, processedImg);

  // Region segmentation using two-pass segmentation algorithm

  // Region Analysis to filter out small regions and get the region of interest (ROI)
  // Assign the ROI to the output parameter for use in feature extraction

  // output = RegionDetect::getROI(processedImg);

  return processedImg; // Return image with detected regions for display/testing(bounding boxes, etc.)
}