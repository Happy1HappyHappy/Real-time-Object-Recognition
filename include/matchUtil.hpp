/*
Claire Liu, Yu-Jing Wei
matchUtil.hpp

Path: include/matchUtil.hpp
Description: Header file for matchUtil.cpp to
             compute matches between feature vectors.
*/

#pragma once // Include guard

#include <vector>
#include "matchResult.hpp"

/*
DBFeature struct represents a feature extracted from a database image.
- featureType: The type of the feature.
- position: The position of the feature in the image.
- metric: The metric used to compare features.
- values: The feature values.
*/
struct DBFeature
{
    std::string featureType;
    std::string position;
    std::string metric;
    std::vector<float> values;
};

/*
MatchUtil class provides static utility functions for comparing and retrieving match results.
- compareMatches(const MatchResult &a, const MatchResult &b): A static function that compares
two MatchResult objects based on their distance values. It returns true if the distance of a
is less than the distance of b, false otherwise.
- getTopNMatches(const std::vector<MatchResult> &results, int N): A static function that takes
a vector of MatchResult objects and an integer N, and returns a vector containing the top N
matches based on the distance values. It also prints the top N matches to the console.
*/
class MatchUtil
{
public:
    static bool compareMatches(const MatchResult &a, const MatchResult &b);
    static std::vector<MatchResult> getTopNMatches(const std::vector<MatchResult> &results, int N);
};