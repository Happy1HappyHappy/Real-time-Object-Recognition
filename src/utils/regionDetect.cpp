/*
  Claire Liu, Yu-Jing Wei
  RegionDetect.cpp
  Path: src/utils/RegionDetect.cpp
  Description: Detects connected regions in binary images and assigns unique labels.
*/

#include "regionDetect.hpp"

#include <cstdint>
#include <vector>

/*
twoPassSegmentation implements the two-pass connected component labeling algorithm.
It takes a binary image as input and produces a label image where each connected
region is assigned a unique integer label. The first pass assigns temporary labels
and records equivalences, while the second pass resolves these equivalences to
produce the final labeled image.
*/
void RegionDetect::twoPassSegmentation(const cv::Mat &src, cv::Mat &dst)
{
    // make sure using 32-bit to avoid overflow
    dst.create(src.size(), CV_32SC1);

    // initialize to 0 for background
    dst.setTo(cv::Scalar(0));
    std::vector<int> parent;
    parent.push_back(0); // Index 0 for background
    int nextLabel = 1;

    // first pass
    for (int y = 0; y < src.rows; ++y)
    {
        for (int x = 0; x < src.cols; ++x)
        {
            if (src.at<uchar>(y, x) > 0)
            {

                // get left and top neighbors
                int leftLabel = (x > 0) ? dst.at<int>(y, x - 1) : 0;
                int topLabel = (y > 0) ? dst.at<int>(y - 1, x) : 0;

                if (leftLabel == 0 && topLabel == 0)
                {
                    dst.at<int>(y, x) = nextLabel;
                    parent.push_back(nextLabel);
                    nextLabel++;
                }
                else if (leftLabel != 0 && topLabel == 0)
                {
                    dst.at<int>(y, x) = leftLabel;
                }
                else if (leftLabel == 0 && topLabel != 0)
                {
                    dst.at<int>(y, x) = topLabel;
                }
                else
                {
                    int minLabel = std::min(leftLabel, topLabel);
                    dst.at<int>(y, x) = minLabel;

                    int root1 = leftLabel;
                    while (parent[root1] != root1)
                        root1 = parent[root1];

                    int root2 = topLabel;
                    while (parent[root2] != root2)
                        root2 = parent[root2];

                    if (root1 != root2)
                    {
                        int minRoot = std::min(root1, root2);
                        int maxRoot = std::max(root1, root2);
                        parent[maxRoot] = minRoot;
                    }
                }
            }
        }
    }
    // update parent
    for (size_t i = 1; i < parent.size(); ++i)
    {
        int root = i;
        while (parent[root] != root)
        {
            root = parent[root];
        }
        parent[i] = root;
    }

    // assign value back to pixels
    for (int y = 0; y < src.rows; ++y)
    {
        for (int x = 0; x < src.cols; ++x)
        {
            int currentLabel = dst.at<int>(y, x);
            if (currentLabel > 0)
            {
                dst.at<int>(y, x) = parent[currentLabel];
            }
        }
    }
}

/*
colorizeRegionLabels takes a label image where each pixel's value corresponds to a region
ID and produces a color visualization. Each unique region ID is assigned a random color,
and the output image is a color representation of the labeled regions. The function uses a
random number generator to create a color palette for the regions, and then maps each pixel
in the label image to its corresponding color in the output image.
*/
cv::Mat RegionDetect::colorizeRegionLabels(const cv::Mat &regionMap32S, uint64_t seed)
{
    CV_Assert(!regionMap32S.empty());
    CV_Assert(regionMap32S.type() == CV_32S);
    // Find the unique region IDs in the labels matrix to determine how many regions to visualize
    double minLabel = 0.0;
    double maxLabel = 0.0;
    cv::minMaxLoc(regionMap32S, &minLabel, &maxLabel);
    // Create a color visualization image initialized to black; each region will be colored based on its label ID
    cv::Mat vis = cv::Mat::zeros(regionMap32S.size(), CV_8UC3);
    if (maxLabel < 1.0)
    {
        return vis;
    }
    // Generate a random color palette for the regions based on the maximum label ID, using the provided seed for reproducibility
    const int maxId = static_cast<int>(maxLabel);
    const uint64_t rngSeed = (seed == 0) ? 0x9E3779B97F4A7C15ULL : seed;
    cv::RNG rng(static_cast<uint64>(rngSeed));
    // Create a color palette where each index corresponds to a region ID, and assign a random color to
    // each region ID (starting from 1, since 0 is background)
    std::vector<cv::Vec3b> palette(static_cast<size_t>(maxId) + 1, cv::Vec3b(0, 0, 0));
    for (int id = 1; id <= maxId; ++id)
    {
        palette[static_cast<size_t>(id)] = cv::Vec3b(
            static_cast<uchar>(rng.uniform(40, 256)),
            static_cast<uchar>(rng.uniform(40, 256)),
            static_cast<uchar>(rng.uniform(40, 256)));
    }
    // Map each pixel in the label image to its corresponding color in the output visualization image
    // based on the region ID, using the generated color palette
    for (int y = 0; y < regionMap32S.rows; ++y)
    {
        const int *src = regionMap32S.ptr<int>(y);
        cv::Vec3b *dst = vis.ptr<cv::Vec3b>(y);
        for (int x = 0; x < regionMap32S.cols; ++x)
        {
            const int id = src[x];
            if (id > 0 && id <= maxId)
            {
                dst[x] = palette[static_cast<size_t>(id)];
            }
        }
    }

    return vis;
}
