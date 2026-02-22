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
    void customDilationErosion(cv::Mat &src, cv::Mat &dst,  int k_size, int e_steps, int d_steps, bool is4Way = false);
    
private:
    // default parameters for morphological filter
    const int DEFAULT_K_SIZE = 5;
    const int DEFAULT_E_STEPS = 2;
    const int DEFAULT_D_STEPS = 2;
    const bool DEFAULT_IS_4WAY = false;

    void dilation(const cv::Mat* src, cv::Mat* dst, int k_size, bool is4Way);
    void erosion(const cv::Mat* src, cv::Mat* dst, int k_size, bool is4Way);
};
