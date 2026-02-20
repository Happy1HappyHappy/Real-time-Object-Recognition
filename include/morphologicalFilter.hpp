/*
  Claire Liu, Yu-Jing Wei
  morphologicalFilter.hpp

  Path: include/morphologicalFilter.hpp
  Description: Header file for morphologicalFilter.cpp to apply morphological filters to images.
*/

#pragma once // Include guard
#include <opencv2/opencv.hpp>

class MorphologicalFilter{
public:
    void defaultDilationErosion(cv::Mat &src, cv::Mat &dst);
    void customDilationErosion(cv::Mat &src, cv::Mat &dst,  int k_size, int e_steps, int d_steps);
    
private:
    // default parameters for morphological filter
    const int DEFAULT_K_SIZE = 3;
    const int DEFAULT_E_STEPS = 1;
    const int DEFAULT_D_STEPS = 1;

    void dilation(const cv::Mat* src, cv::Mat* dst, int k_size);
    void erosion(const cv::Mat* src, cv::Mat* dst, int k_size);
};