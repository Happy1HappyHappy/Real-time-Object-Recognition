/*
Claire Liu, Yu-Jing Wei
DistanceTransform.hpp

Path: include/distanceTransform.hpp
Description: Distance transform utilities (grassfire / connected components).
*/

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

/*
DistanceTransform class to handle distance transform operations
such as grassfire and connected components.
*/
class DistanceTransform
{
public:
    DistanceTransform();
    ~DistanceTransform();
    static void grassfire(cv::Mat &binaryImage, cv::Mat &regionMap);

private:
};
