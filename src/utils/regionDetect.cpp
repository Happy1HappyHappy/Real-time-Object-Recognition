# include "regionDetect.hpp"

void RegionDetect::twoPassSegmentation(const cv::Mat &binaryImage, cv::Mat &regionMap)
{
    CV_Assert(!binaryImage.empty());
    CV_Assert(binaryImage.type() == CV_8U);

    cv::Mat bin = binaryImage.clone();
    if (bin.channels() != 1)
    {
        cv::cvtColor(bin, bin, cv::COLOR_BGR2GRAY);
    std::vector<int> parent;
    parent.push_back(0); // Index 0 for background
    int nextLabel = 1;

    // first pass
    for (int y = 0; y < binaryImage.rows; ++y)
    {
        for (int x = 0; x < binaryImage.cols; ++x)
        {
            if (binaryImage.at<uchar>(y, x) > 0)
            {

                // get left and top neighbors
                int leftLabel = (x > 0) ? regionMap.at<int>(y, x - 1) : 0;
                int topLabel = (y > 0) ? regionMap.at<int>(y - 1, x) : 0;

                if (leftLabel == 0 && topLabel == 0)
                {
                    regionMap.at<int>(y, x) = nextLabel;
                    parent.push_back(nextLabel);
                    nextLabel++;
                }
                else if (leftLabel != 0 && topLabel == 0)
                {
                    regionMap.at<int>(y, x) = leftLabel;
                }
                else if (leftLabel == 0 && topLabel != 0)
                {
                    regionMap.at<int>(y, x) = topLabel;
                }
                else
                {
                    int minLabel = std::min(leftLabel, topLabel);
                    regionMap.at<int>(y, x) = minLabel;

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

    cv::connectedComponents(bin, regionMap, 8, CV_32S);
      // update parent
    for (size_t i = 1; i < parent.size(); ++i) {
        int root = i;
        while (parent[root] != root) {
            root = parent[root];
        }
        parent[i] = root; 
    }

    // assign value back to pixels
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            int currentLabel = dst.at<int>(y, x);
            if (currentLabel > 0) {
                dst.at<int>(y, x) = parent[currentLabel]; 
            }
        }
    }
}

cv::Mat RegionDetect::colorizeRegionLabels(const cv::Mat &regionMap32S, uint64_t seed)
{
    CV_Assert(!regionMap32S.empty());
    CV_Assert(regionMap32S.type() == CV_32S);

    double minLabel = 0.0, maxLabel = 0.0;
    cv::minMaxLoc(regionMap32S, &minLabel, &maxLabel);

    cv::Mat vis = cv::Mat::zeros(regionMap32S.size(), CV_8UC3);
    if (maxLabel < 1.0)
    {
        return vis;
    }

    if (seed == 0)
    {
        seed = static_cast<uint64_t>(cv::getTickCount());
    }
    cv::RNG rng(seed); // Random Number Generator
    std::vector<cv::Vec3b> palette(static_cast<size_t>(maxLabel) + 1, cv::Vec3b(0, 0, 0));
    for (int label = 1; label <= static_cast<int>(maxLabel); ++label)
    {
        palette[static_cast<size_t>(label)] = cv::Vec3b(
            static_cast<uchar>(rng.uniform(40, 255)),
            static_cast<uchar>(rng.uniform(40, 255)),
            static_cast<uchar>(rng.uniform(40, 255)));
    }

    for (int y = 0; y < regionMap32S.rows; ++y)
    {
        const int *srcRow = regionMap32S.ptr<int>(y);
        cv::Vec3b *dstRow = vis.ptr<cv::Vec3b>(y);
        for (int x = 0; x < regionMap32S.cols; ++x)
        {
            const int id = srcRow[x];
            if (id > 0 && id <= static_cast<int>(maxLabel))
            {
                dstRow[x] = palette[static_cast<size_t>(id)];
            }
        }
    }
    return vis;

}
