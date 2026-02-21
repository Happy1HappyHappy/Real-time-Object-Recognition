/*
Claire Liu, Yu-Jing Wei
featureMatcher.cpp

Path: src/online/featureMatcher.cpp
Description: Matches features from a query image to a database of feature
vectors.
*/

#include "distanceMetrics.hpp"
#include "featureMatcher.hpp"
#include "metricFactory.hpp"
#include "readFiles.hpp"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <vector>

bool FeatureMatcher::match(
    const std::vector<float> &targetFeatures,
    const std::string &dbPath,
    MetricType metricType,
    MatchResult &bestMatch)
{
    std::vector<std::string> dbLabels;
    std::vector<std::vector<float>> dbData;
    if (ReadFiles::readFeaturesFromCSV(dbPath.c_str(), dbLabels, dbData) != 0 || dbData.empty())
    {
        return false;
    }

    auto distanceMetric = MetricFactory::create(metricType);
    if (!distanceMetric)
    {
        return false;
    }

    bool found = false;
    MatchResult best{"", 0.0f};
    for (size_t i = 0; i < dbData.size(); ++i)
    {
        const float distance = distanceMetric->compute(targetFeatures, dbData[i]);
        if (!std::isfinite(distance))
        {
            continue;
        }
        if (!found || distance < best.distance)
        {
            found = true;
            best.filename = dbLabels[i];
            best.distance = distance;
        }
    }

    if (!found)
    {
        return false;
    }

    bestMatch = best;
    return true;
}
