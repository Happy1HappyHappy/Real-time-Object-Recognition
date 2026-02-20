/*
Claire Liu, Yu-Jing Wei
MatchResult.hpp

Path: include/matchResult.hpp
Description: Header file for MatchResult struct to store matching results.
*/

#pragma once
#include <string>

/*
MatchResult struct is a simple data structure that holds the results of a matching
operation between a query image and a database image. It contains:
- filename: A string that stores the filename of the matched image from the database.
    This allows users to identify which image in the database is the closest match to
    the query image based on the computed distance.
- distance: A float that represents the computed distance between the feature vector
    of the query image and the feature vector of the matched image in the database.
*/
struct MatchResult
{
    // The filename of the matched image
    std::string filename;
    // The distance value representing the similarity between the query image and the matched image
    float distance;
};
