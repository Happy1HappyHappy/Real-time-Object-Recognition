/*
  Claire Liu, Yu-Jing Wei
  matchUtil.cpp

  Path: src/utils/matchUtil.cpp
  Description: Implements utility functions for matching feature vectors.
  */

#include "matchUtil.hpp"
#include "matchResult.hpp"
#include <algorithm>
#include <vector>

/*
compareMatches is a helper function used to sort MatchResult objects
based on their distance values in ascending order.
- @param a: The first MatchResult object to compare.
- @param b: The second MatchResult object to compare.
- @return: true if the distance of a is less than the distance of b, false otherwise.
*/
bool MatchUtil::compareMatches(const MatchResult &a, const MatchResult &b)
{
    return a.distance < b.distance;
}

/*
getTopNMatches is a utility function that takes a vector of MatchResult objects and an
integer N, and returns a vector containing the top N matches based on the distance values.
- @param results: A vector of MatchResult objects that have been sorted by distance.
- @param N: The number of top matches to return.
- @return: A vector of MatchResult objects containing the top N matches from the input vector.
*/
std::vector<MatchResult> MatchUtil::getTopNMatches(const std::vector<MatchResult> &results, int N)
{
    std::vector<MatchResult> topMatches;
    // Print top N matches
    printf("Top %d matches:\n", N);
    for (int i = 0; i < N && i < (int)results.size(); ++i)
    {
        printf("%s %f\n", results[i].filename.c_str(), results[i].distance);
        topMatches.push_back(results[i]);
    }
    return topMatches;
}
