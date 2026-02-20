/*
  Claire Liu, Yu-Jing Wei
  preProcessor.cpp

  Path: src/utils/preProcessor.cpp
  Description: Pre-processes images before feature extraction.
*/

#include "preProcessor.hpp"
#include "regionDetect.hpp"
#include "thresholding.hpp"
#include "morphologicalFilter.hpp"
#include <opencv2/opencv.hpp>

cv::Mat PreProcessor::process(const cv::Mat &input)
{
    // cv thresholding to get a binary image
    cv::Mat gray;
    // binary image
    cv::Mat binary;
    // cleaned binary image
    cv::Mat cleanedBinary;
    // region image
    cv::Mat region;

    // ultimate output(not yet implemented)
    cv::Mat output;
    
    // convert to grey scale
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    cv::imshow("1. Gray Image", gray);
    cv::waitKey(0);
    
    // apply thresholding to get a binary image
    Threadsholding::dynamicThreadsHold(gray, binary);
    cv::imshow("2. Binary Image", binary);
    cv::waitKey(0);

    // apply morphological filter to remove noise
    MorphologicalFilter myFilter;
    myFilter.defaultDilationErosion(binary, cleanedBinary);
    cv::imshow("3. Cleaned Binary Image", cleanedBinary);
    cv::waitKey(0);
    
    // Region detection using grassfire algorithm
    RegionDetect::grassfire(cleanedBinary, region);
    
    // Display region
    cv::Mat regionVis;
    cv::normalize(region, regionVis, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("4. Region Map", regionVis);
    cv::waitKey(0);

    return output;
}