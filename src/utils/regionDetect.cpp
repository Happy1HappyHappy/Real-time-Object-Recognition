# include "regionDetect.hpp"

void RegionDetect::twoPassSegmentation(cv::Mat &src, cv::Mat &dst){
    // make sure using 32-bit to avoid overflow
    dst.create(src.size(), CV_32SC1);
    
    // initialize to 0 for background
    dst.setTo(cv::Scalar(0)); 

    std::vector<int> parent;
    parent.push_back(0); // Index 0 for background
    int nextLabel = 1;

    // first pass 
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            if (src.at<uchar>(y, x) > 0) {
                
                // get left and top neighbors
                int leftLabel = (x > 0) ? dst.at<int>(y, x - 1) : 0;
                int topLabel  = (y > 0) ? dst.at<int>(y - 1, x) : 0;

                if (leftLabel == 0 && topLabel == 0) {
                    dst.at<int>(y, x) = nextLabel;
                    parent.push_back(nextLabel); 
                    nextLabel++;
                } 
                else if (leftLabel != 0 && topLabel == 0) {
                    dst.at<int>(y, x) = leftLabel;
                } 
                else if (leftLabel == 0 && topLabel != 0) {
                    dst.at<int>(y, x) = topLabel;
                } 
                else {
                    int minLabel = std::min(leftLabel, topLabel);
                    dst.at<int>(y, x) = minLabel;

                    int root1 = leftLabel;
                    while (parent[root1] != root1) root1 = parent[root1];
                    
                    int root2 = topLabel;
                    while (parent[root2] != root2) root2 = parent[root2];

                    if (root1 != root2) {
                        int minRoot = std::min(root1, root2);
                        int maxRoot = std::max(root1, root2);
                        parent[maxRoot] = minRoot;
                    }
                }
            }
        }
    }

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
