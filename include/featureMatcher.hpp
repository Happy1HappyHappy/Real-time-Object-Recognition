/*
Claire Liu, Yu-Jing Wei
featureMatcher.hpp

Path: include/featureMatcher.hpp
Description: Header file for featureMatcher.cpp to
             match features between a query image and a database.
*/

#pragma once // Include guard

#include "extractorFactory.hpp"
#include "metricFactory.hpp"
#include "matchResult.hpp"

/*
FeatureMatcher class provides a static method to match features
between a query image and a database of images.
*/
class FeatureMatcher
{
public:
    static bool match(
        const std::vector<float> &targetFeatures,
        const std::string &dbPath,
        MetricType metricType,
        MatchResult &bestMatch);
};
