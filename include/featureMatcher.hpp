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

class FeatureMatcher
{
public:
    static int match(
        std::vector<float> targetFeatures,
        std::string dbPath,
        MetricType metricType);
};