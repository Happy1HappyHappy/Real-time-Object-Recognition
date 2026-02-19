/*
Claire Liu, Yu-Jing Wei
Filters.hpp

Path: project1/include/utils/Filters.hpp
Description: Declares image filtering functions (e.g., greyscale, blur, etc.).
*/

#pragma once // Include guard

#include <opencv2/opencv.hpp>

/*
Filters class to apply various image filtering operations.
public:
- Sobel X 3x3: Applies a 3x3 Sobel filter in the X direction.
- Sobel Y 3x3: Applies a 3x3 Sobel filter in the Y direction.
- Magnitude: Computes the gradient magnitude from Sobel X and Y.
- Face Detection: Detects faces in an image.
- Gabor Filter: Applies a Gabor filter to an image.
- CIELab: Converts an image from BGR to CIELab color space.
- Convolution: Applies a custom convolution kernel to an image.
*/
class Filters
{
public:
    static int sobelX3x3(cv::Mat &src, cv::Mat &dst);
    static int sobelY3x3(cv::Mat &src, cv::Mat &dst);
    static int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
    static int faceDetect(cv::Mat &src, cv::Mat &dst, cv::Rect &last);
    static int CIELab(cv::Mat &src, cv::Mat &dst);
    static int gabor(cv::Mat &src, cv::Mat &dst);
    static int convolve(cv::Mat &src, cv::Mat &dst, int *kernel1, int *kernel2, int kSize, int kSum);

private:
    static int GaborBankGenerator(std::vector<cv::Mat> *filters);
};
