#include "regionDetect.hpp"
#include <opencv2/opencv.hpp>

// Grassfire algorithm.
// Input: binary image (CV_8UC1, 0 for background, 255 for foreground).
// Output: region map (CV_8UC1, 0 for background, 1 for region 1, 2 for region 2, ...).
void RegionDetect::grassfire(cv::Mat &src, cv::Mat &regionMap)
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

    // show result, can skip
    cv::Mat vis;
    vis = 255 * regionMap / maxVal;
    cv::imshow("pass1_top-down", vis);
    cv::waitKey(0);

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

    // show result, can skip
    vis = 255 * regionMap / maxVal;
    cv::imshow("pass2_final", vis);
    cv::waitKey(0);
}

void RegionDetect::twoSegmentation(cv::Mat &src, cv::Mat &regionMap)
{
    return;
}
