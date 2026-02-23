/*
  Claire Liu, Yu-Jing Wei
  morphologicalFilter.cpp

  Path: src/utils/morphologicalFilter.cpp
  Description: Implements morphological filtering operations such as dilation and erosion.
*/

#include "morphologicalFilter.hpp"
#include <opencv2/opencv.hpp>

/*
defaultDilationErosion applies a standard morphological filter with predefined parameters to the input image.
It performs a series of erosions followed by dilations to clean up the binary image.
*/
void MorphologicalFilter::defaultDilationErosion(cv::Mat &src, cv::Mat &dst)
{
    customDilationErosion(src, dst, DEFAULT_K_SIZE, DEFAULT_E_STEPS, DEFAULT_D_STEPS, DEFAULT_IS_4WAY);
}

/**
 * Custom morphological filter with adjustable kernel size and iterations.
 * @param src Input binary image.
 * @param dst Output processed image.
 * @param k_size Size of the structuring element (e.g., 3, 5, 7).
 * @param e_steps Number of erosion iterations.
 * @param d_steps Number of dilation iterations.
 * @param is4Way If true, uses 4-way connectivity (cross). Otherwise, uses 8-way (square).
 */
void MorphologicalFilter::customDilationErosion(cv::Mat &src, cv::Mat &dst, int k_size, int e_steps, int d_steps, bool is4Way)
{
    // We use a temporary matrix to hold intermediate results during iterations.
    cv::Mat current_stage = src.clone();
    cv::Mat next_stage;

    // Perform Erosion
    for (int i = 0; i < e_steps; i++)
    {
        erosion(&current_stage, &next_stage, k_size, is4Way);
        // Update current_stage with the result for the next iteration.
        current_stage = next_stage.clone();
    }

    // Perform Dilation
    for (int i = 0; i < d_steps; i++)
    {
        dilation(&current_stage, &next_stage, k_size, is4Way);
        current_stage = next_stage.clone();
    }

    // Assign the final processed result to the destination matrix.
    dst = current_stage;
}

/*
dilation applies dilation to the input binary image using a specified kernel size and connectivity.
*/
void MorphologicalFilter::dilation(const cv::Mat *src, cv::Mat *dst, int k_size, bool is4Way)
{
    int h = src->rows;
    int w = src->cols;
    int pad = k_size / 2;

    *dst = cv::Mat::zeros(h, w, CV_8UC1);

    cv::Mat paddedImg;
    // padding the image to avoid boundary issues
    copyMakeBorder(*src, paddedImg, pad, pad, pad, pad, cv::BORDER_CONSTANT, cv::Scalar(0));

    // iterate through the image source image from (0,0)
    // use padded image for reference to avoid boundary issues
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            bool any_white = false;
            for (int ki = 0; ki < k_size; ki++)
            {
                for (int kj = 0; kj < k_size; kj++)
                {
                    // Check connectivity (4-way: skip corners)
                    if (is4Way && ki != pad && kj != pad)
                    {
                        continue;
                    }

                    if (paddedImg.at<uchar>(i + ki, j + kj) == 255)
                    {
                        any_white = true;
                        break;
                    }
                }
                if (any_white)
                    break;
            }
            dst->at<uchar>(i, j) = any_white ? 255 : 0;
        }
    }
}

/*
erosion applies erosion to the input binary image using a specified kernel size and connectivity.
*/
void MorphologicalFilter::erosion(const cv::Mat *src, cv::Mat *dst, int k_size, bool is4Way)
{
    int h = src->rows;
    int w = src->cols;
    int pad = k_size / 2;

    *dst = cv::Mat::zeros(h, w, CV_8UC1);

    cv::Mat paddedImg;
    // padding the image to avoid boundary issues
    cv::copyMakeBorder(*src, paddedImg, pad, pad, pad, pad, cv::BORDER_CONSTANT, cv::Scalar(255));

    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            bool all_white = true;
            for (int ki = 0; ki < k_size; ki++)
            {
                for (int kj = 0; kj < k_size; kj++)
                {
                    // Check connectivity (4-way: skip corners)
                    if (is4Way && ki != pad && kj != pad)
                    {
                        continue;
                    }

                    if (paddedImg.at<uchar>(i + ki, j + kj) == 0)
                    {
                        all_white = false;
                        break;
                    }
                }
                if (!all_white)
                    break;
            }
            dst->at<uchar>(i, j) = all_white ? 255 : 0;
        }
    }
}
