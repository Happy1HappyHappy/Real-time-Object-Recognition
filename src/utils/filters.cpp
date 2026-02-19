/*
Claire Liu, Yu-Jing Wei
Filters.cpp

Path: src/utils/Filters.cpp
Description: Filters implementation file defining image filtering functions.
*/

#include "filters.hpp"
#include "faceDetect.hpp"
#include <opencv2/opencv.hpp>

/*
Applies the Sobel X filter to the source image.
- @param src The source image.
- @param dst The destination image where the result will be stored.
- @return 0 on success, -1 on failure (e.g., if the source image is empty).
*/
int Filters::sobelX3x3(cv::Mat &src, cv::Mat &dst)
{
    // This function applies the Sobel X filter to the source image
    // and stores the result in the destination image.
    // src: source image
    // dst: destination image

    // check for empty source images
    if (src.empty())
        return -1;
    // define the 1x3 Sobel separable kernels
    int kernelXH[3] = {-1, 0, 1}; // Horizontal kernel
    int kernelXV[3] = {1, 2, 1};  // Vertical kernel

    // Convert to grayscale first
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);
    // Convolve
    Filters::convolve(grey, dst, kernelXH, kernelXV, 3, 0);
    // convert to absolute values and 8U type
    cv::convertScaleAbs(dst, dst);
    return 0;
}

/*
Applies the Sobel Y filter to the source image.
- @param src The source image.
- @param dst The destination image where the result will be stored.
- @return 0 on success, -1 on failure (e.g., if the source image is empty).
*/
int Filters::sobelY3x3(cv::Mat &src, cv::Mat &dst)
{
    // This function applies the Sobel Y filter to the source image
    // and stores the result in the destination image.
    // src: source image
    // dst: destination image

    // check for empty source images
    if (src.empty())
        return -1;
    // define the 1x3 Sobel separable kernels
    int kernelYH[3] = {1, 2, 1};  // Horizontal kernel
    int kernelYV[3] = {-1, 0, 1}; // Vertical kernel

    // greyscale first
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);
    // Convolve
    Filters::convolve(grey, dst, kernelYH, kernelYV, 3, 0);
    // convert to absolute values and 8U type
    cv::convertScaleAbs(dst, dst);

    return 0;
}

/*
Computes the magnitude of two images sx and sy.
- @param sx The Sobel X image.
- @param sy The Sobel Y image.
- @param dst The output magnitude image.
- @return 0 on success, -1 on failure (e.g., if the source images are empty).
*/
int Filters::magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst)
{
    // This function computes the magnitude of two images sx and sy
    // and stores the result in dst.
    // sx: Sobel X image
    // sy: Sobel Y image
    // dst: output magnitude image

    // check for empty source images
    if (sx.empty() || sy.empty())
        return -1;
    // allocate dst if empty
    if (dst.empty())
        dst.create(sx.size(), sx.type());

    // compute the magnitude of the two images
    for (int i = 0; i < sx.rows; i++)
    {
        // Use pointers for faster access
        // Vec3s for possible negative values from Sobel
        cv::Vec3b *sxRow = sx.ptr<cv::Vec3b>(i);
        cv::Vec3b *syRow = sy.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        // iterate each column
        for (int j = 0; j < sx.cols; j++)
        {
            // iterate each colorchannel
            for (int c = 0; c < 3; c++)
            {
                int sx_val = sxRow[j][c];
                int sy_val = syRow[j][c];

                // compute magnitude
                int magnitude = (int)sqrt(sx_val * sx_val + sy_val * sy_val);

                // clamp to [0,255] and assign to dst
                dstRow[j][c] = cv::saturate_cast<uchar>(magnitude);
            }
        }
    }

    return 0;
}

/*
Detects faces in the source image and draws rectangles around them in the destination image.
- @param src The source image.
- @param dst The destination image where the result will be stored.
- @param last The last detected face rectangle for smoothing.
- @return 0 on success, -1 on failure (e.g., if the source image is empty).
*/
int Filters::faceDetect(cv::Mat &src, cv::Mat &dst, cv::Rect &last)
{
    // This function detects faces in the source image and draws rectangles around them in the destination image
    // src: source image
    // dst: destination image

    // check for empty source images
    if (src.empty())
        return -1;

    // copy src to dst
    src.copyTo(dst);

    std::vector<cv::Rect> faces;                 // vector to hold detected faces
    cv::Mat grey;                                // grayscale image
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY); // convert to greyscale

    // detect faces
    detectFaces(grey, faces);

    // add a little smoothing by averaging the last two detections
    if (faces.size() > 0)
    {
        if (last.area() == 0)
        {
            last = faces[0]; // initialize last if it's the first detection
        }
        else
        {
            // smooth by averaging with last
            last.x = (faces[0].x + last.x) / 2;
            last.y = (faces[0].y + last.y) / 2;
            last.width = (faces[0].width + last.width) / 2;
            last.height = (faces[0].height + last.height) / 2;
        }
    }
    // draw boxes around the faces
    drawBoxes(dst, faces, 0, 1.0);

    // check if dst is empty
    if (dst.empty())
        return -1;

    return 0;
}

int Filters::CIELab(cv::Mat &src, cv::Mat &dst)
{
    // This function converts an image from BGR to CIELab color space
    dst.create(src.size(), CV_32FC3);

    for (int i = 0; i < src.rows; i++)
    {
        const cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3f *dstRow = dst.ptr<cv::Vec3f>(i);
        for (int j = 0; j < src.cols; j++)
        {
            float b_norm = ptr[j][0] / 255.0f;
            float g_norm = ptr[j][1] / 255.0f;
            float r_norm = ptr[j][2] / 255.0f;

            // Inverse Gamma Correction
            float r_lin = (r_norm > 0.04045f) ? pow((r_norm + 0.055f) / 1.055f, 2.4f) : r_norm / 12.92f;
            float g_lin = (g_norm > 0.04045f) ? pow((g_norm + 0.055f) / 1.055f, 2.4f) : g_norm / 12.92f;
            float b_lin = (b_norm > 0.04045f) ? pow((b_norm + 0.055f) / 1.055f, 2.4f) : b_norm / 12.92f;

            // Convert to XYZ color space
            // X: response corresponding roughly to red-green perception
            // Y: luminance (perceived brightness)
            // Z: response corresponding roughly to blue perception
            float X = r_lin * 0.4124f + g_lin * 0.3576f + b_lin * 0.1805f;
            float Y = r_lin * 0.2126f + g_lin * 0.7152f + b_lin * 0.0722f;
            float Z = r_lin * 0.0193f + g_lin * 0.1192f + b_lin * 0.9505f;

            // Normalize XYZ by the reference white (D65)
            // Using D65 reference
            float x = X / 0.95047f;
            float y = Y / 1.00000f;
            float z = Z / 1.08883f;

            // Lab non-linear transform
            float fx = (x > 0.008856f) ? cbrt(x) : (7.787f * x + 16.0f / 116.0f);
            float fy = (y > 0.008856f) ? cbrt(y) : (7.787f * y + 16.0f / 116.0f);
            float fz = (z > 0.008856f) ? cbrt(z) : (7.787f * z + 16.0f / 116.0f);

            // Convert to Lab
            float L = 116.0f * fy - 16.0f;
            float a = 500.0f * (fx - fy);
            float b = 200.0f * (fy - fz);

            // set the new Lab values to dst
            dstRow[j] = cv::Vec3f(L, a, b);
        }
    }
    return 0; // success
}

int Filters::GaborBankGenerator(std::vector<cv::Mat> *filters)
{
    int ksize = 31;      // kernel size
    double sigma = 4.0;  // Gaussian sigma
    double lambd = 10.0; // wavelength
    double gamma = 0.5;  // Aspect ratio
    double psi = 0;      // Phase offset

    double thetas[] = {0, CV_PI / 4, CV_PI / 2, CV_PI * 3 / 4};

    for (double theta : thetas)
    {
        cv::Mat kernel = getGaborKernel(cv::Size(ksize, ksize), sigma, theta, lambd, gamma, psi, CV_32F);
        filters->push_back(kernel);
    }
    return 0;
}

int Filters::gabor(cv::Mat &src, cv::Mat &dst)
{
    cv::Mat gray;

    // Check if image is in gray
    if (src.channels() > 1)
    {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = src;
    }

    // Get Gabor filters
    std::vector<cv::Mat> gaborFilters;
    GaborBankGenerator(&gaborFilters);

    // Process each Gabor filter
    for (const cv::Mat &kernel : gaborFilters)
    {
        cv::Mat fimg;

        // CV32F to prevent overflow during convolution
        cv::filter2D(gray, fimg, CV_32F, kernel);

        // Transfer to 8-bit unsigned for histogram calculation
        cv::Mat fimg_8u;
        cv::convertScaleAbs(fimg, fimg_8u);
    }
    return 0; // success
}

/*
Convolves the source image with separable kernels.
- @param src The source image.
- @param dst The destination image where the result will be stored.
- @param kernel1 The horizontal 1D kernel.
- @param kernel2 The vertical 1D kernel.
- @param kSize The size of the kernels (assumed to be odd).
- @param kSum The sum of the kernel weights for normalization (0 if no normalization).
- @return 0 on success, -1 on failure (e.g., if the source image is empty).
*/
int Filters::convolve(cv::Mat &src, cv::Mat &dst, int *kernel1, int *kernel2, int kSize, int kSum)
{
    // This is a function only for convolving an image with separable kernel
    // src: source image
    // dst: destination image
    // kernel1: horizontal 1D kernel
    // kernel2: vertical 1D kernel
    // kSize: size of the kernels (assumed to be odd)
    // kSum: sum of the kernel weights for normalization (0 if no normalization)

    // check for empty source images
    if (src.empty() || kernel1 == nullptr || kernel2 == nullptr)
        return -1;

    // allocate dst if empty
    if (dst.empty())
        dst.create(src.size(), src.type());

    // ensure src is of type CV_8UC1(greyscale) or CV_8UC3(3 color channels)
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3);

    int channels = src.channels(); // number of color channels
    int rows = src.rows;           // image row dimensions
    int cols = src.cols;           // image col dimensions
    const int kHalf = kSize / 2;   // padding size

    // intermediate buffer (signed!)
    cv::Mat tmp(rows, cols, CV_MAKETYPE(CV_16S, channels));

    // output (signed, keep direction info)
    dst.create(rows, cols, CV_MAKETYPE(CV_16S, channels));

    /*  Horizontal Pass*/
    if (channels == 1) // greyscale
    {
        for (int i = 0; i < rows; i++)
        {
            const uchar *srcRow = src.ptr<uchar>(i);
            short *tmpRow = tmp.ptr<short>(i);

            for (int j = 0; j < cols; j++)
            {
                int sum = 0;
                for (int k = -kHalf; k <= kHalf; k++)
                {
                    int col = std::clamp(j + k, 0, cols - 1);
                    sum += srcRow[col] * kernel1[k + kHalf];
                }
                if (kSum)
                    sum /= kSum;
                tmpRow[j] = sum;
            }
        }
    }
    else // 3 channels
    {
        for (int i = 0; i < rows; i++)
        {
            const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
            cv::Vec3s *tmpRow = tmp.ptr<cv::Vec3s>(i);

            for (int j = 0; j < cols; j++)
            {
                for (int c = 0; c < 3; c++)
                {
                    int sum = 0;
                    for (int k = -kHalf; k <= kHalf; k++)
                    {
                        int col = std::clamp(j + k, 0, cols - 1);
                        sum += srcRow[col][c] * kernel1[k + kHalf];
                    }
                    if (kSum)
                        sum /= kSum;
                    tmpRow[j][c] = sum;
                }
            }
        }
    }

    /* Vertical Pass */
    if (channels == 1) // greyscale
    {
        for (int i = 0; i < rows; i++)
        {
            short *dstRow = dst.ptr<short>(i);

            for (int j = 0; j < cols; j++)
            {
                int sum = 0;
                for (int k = -kHalf; k <= kHalf; k++)
                {
                    int row = std::clamp(i + k, 0, rows - 1);
                    sum += tmp.ptr<short>(row)[j] * kernel2[k + kHalf];
                }
                if (kSum)
                    sum /= kSum;
                dstRow[j] = sum;
            }
        }
    }
    else // 3 channels
    {
        for (int i = 0; i < rows; i++)
        {
            cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(i);

            for (int j = 0; j < cols; j++)
            {
                for (int c = 0; c < 3; c++)
                {
                    int sum = 0;
                    for (int k = -kHalf; k <= kHalf; k++)
                    {
                        int row = std::clamp(i + k, 0, rows - 1);
                        sum += tmp.ptr<cv::Vec3s>(row)[j][c] * kernel2[k + kHalf];
                    }
                    if (kSum)
                        sum /= kSum;
                    dstRow[j][c] = sum;
                }
            }
        }
    }
    return 0;
}
