/*
  Claire Liu, Yu-Jing Wei
  distanceTransform.cpp

  Path: src/utils/distanceTransform.cpp
  Description: Implements distance transform algorithms for image processing.
*/

#include "distanceTransform.hpp"
#include <opencv2/opencv.hpp>

/*
grassfire implements the grassfire distance transform algorithm on a binary image.
It computes the distance of each foreground pixel to the nearest background pixel and stores the result in regionMap.
input:
- src: a binary image (CV_8UC1) where foreground pixels are non-zero and background pixels are zero.
output:
- regionMap: a CV_8UC1 image where each pixel value represents the distance to the nearest background pixel.
*/
void DistanceTransform::grassfire(cv::Mat &src, cv::Mat &regionMap)
{
    int maxVal = 1;

    // output image
    regionMap = cv::Mat::zeros(src.size(), CV_8UC1);

    // top-down
    for (int i = 1; i < src.rows - 1; i++)
    {
        uchar *ptr = src.ptr<uchar>(i);
        uchar *uptr = regionMap.ptr<uchar>(i - 1);
        uchar *destPtr = regionMap.ptr<uchar>(i);

        for (int j = 1; j < src.cols - 1; j++)
        {
            // if foreground
            if (ptr[j] > 0)
            {
                int up = uptr[j];
                int left = destPtr[j - 1];
                int cur = up < left ? up : left;

                destPtr[j] = cur + 1;
                if (destPtr[j] > maxVal)
                    maxVal = destPtr[j];
            }
        }
    }

    // bottom-up
    for (int i = src.rows - 2; i > 0; i--)
    {
        uchar *srcPtr = src.ptr<uchar>(i);
        uchar *ptr = regionMap.ptr<uchar>(i);
        uchar *dptr = regionMap.ptr<uchar>(i + 1);

        for (int j = src.cols - 2; j > 0; j--)
        {
            if (ptr[j] > 0)
            {
                int down = dptr[j];
                int right = ptr[j + 1];
                int cur = down < right ? down : right;

                int newVal = cur + 1;
                if (newVal < ptr[j])
                {
                    ptr[j] = (uchar)newVal;
                }
                if (ptr[j] > maxVal)
                    maxVal = ptr[j];
            }
        }
    }
}
