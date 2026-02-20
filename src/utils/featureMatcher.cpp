/*
Claire Liu, Yu-Jing Wei
featureMatcher.cpp

Path: src/online/featureMatcher.cpp
Description: Matches features from a query image to a database of feature
vectors.
*/

#include "distanceMetrics.hpp"
#include "extractorFactory.hpp"
#include "extractor.hpp"
#include "featureMatcher.hpp"
#include "matchResult.hpp"
#include "matchUtil.hpp"
#include "metricFactory.hpp"
#include "readFiles.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

int FeatureMatcher::match(
    std::vector<float> targetFeatures,
    std::string dbPath,
    MetricType metricType)
{

  // Initialize data structures to accumulate distances and track seen images
  std::unordered_map<std::string, float> totalDistance;
  std::unordered_map<std::string, bool> seenAny;

  // Load database feature vectors and filenames from the CSV file
  std::vector<std::string> dbFilenames;   // database to save image filenames
  std::vector<std::vector<float>> dbData; // database to save feature vectors
  ReadFiles::readFeaturesFromCSV(dbPath.c_str(), dbFilenames, dbData);

  if (dbData.empty())
  {
    printf("Warning: DB is empty: %s\n", dbPath.c_str());
    return -1;
  }
  // Create the appropriate distance metric based on the specified metric type
  auto distanceMetric = MetricFactory::create(metricType);
  if (!distanceMetric)
  {
    printf("Error: invalid metric for db entry. db='%s'\n\n", dbPath.c_str());
    return -1;
  }

  // Create the appropriate distance metric based on the specified metric type
  printf("Distance metric: %s\n",
         MetricFactory::metricTypeToString(metricType).c_str());
  printf("--------------------\n");
  // Compute distances between the target features and each database feature
  // vector
  for (size_t i = 0; i < dbData.size(); ++i)
  {
    float d = distanceMetric->compute(targetFeatures, dbData[i]);
    // Accumulate the weighted distance for this database entry
    totalDistance[dbFilenames[i]] += d;
    // Mark this image as seen
    seenAny[dbFilenames[i]] = true;
  }

  // Convert the accumulated distances into a vector of MatchResult objects
  std::vector<MatchResult> results;
  results.reserve(totalDistance.size());

  // Convert the accumulated distances into MatchResult objects
  for (const auto &kv : totalDistance)
  {
    if (!seenAny[kv.first])
      continue;
    MatchResult res;
    res.filename = kv.first;
    res.distance = kv.second;
    results.push_back(res);
  }

  if (results.empty())
  {
    printf("No matches (check DBs / feature extraction).\n");
    return 0;
  }
  // Sort the results by distance
  std::sort(results.begin(), results.end(), MatchUtil::compareMatches);
  std::vector<MatchResult> topMatches =
      MatchUtil::getTopNMatches(results, 1);

  return (0); // Success
}