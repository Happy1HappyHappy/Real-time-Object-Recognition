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
#include <limits>
#include <iostream>
#include <vector>
#include <filesystem>
#include <unordered_map>

// namespace for internal helper functions and caching structures related to feature matching
namespace
{
    /*
    CachedFeatureDb stores the contents and metadata of a feature database CSV file to avoid
    redundant disk reads and computations for matching
    */
    struct CachedFeatureDb
    {
        std::vector<std::string> labels;
        std::vector<std::vector<float>> data;
        std::filesystem::file_time_type lastWriteTime{};
        std::uintmax_t fileSize = 0;
        bool loaded = false;
    };

    // Global cache mapping database file paths to their cached contents and metadata
    std::unordered_map<std::string, CachedFeatureDb> gDbCache;

    /*
    Loads a cached feature database from the given path. If the database is not already cached
    or has been modified since the last load, it reads the database from disk and updates the cache.
    */
    const CachedFeatureDb *loadCachedDb(const std::string &dbPath)
    {
        if (dbPath.empty())
        {
            return nullptr;
        }
        auto it = gDbCache.find(dbPath);
        bool needsReload = (it == gDbCache.end()) || (!it->second.loaded);

        // time settings for cache validation
        std::error_code timeEc;
        std::error_code sizeEc;
        const auto nowWriteTime = std::filesystem::last_write_time(dbPath, timeEc);
        const auto nowFileSize = std::filesystem::file_size(dbPath, sizeEc);

        if (!needsReload && !timeEc && !sizeEc)
        {
            if (it->second.lastWriteTime != nowWriteTime || it->second.fileSize != nowFileSize)
            {
                needsReload = true;
            }
        }
        // If we need to reload (either not cached or file has changed), read the database from disk and update the cache
        if (needsReload)
        {
            std::vector<std::string> dbLabels;
            std::vector<std::vector<float>> dbData;
            if (ReadFiles::readFeaturesFromCSV(dbPath.c_str(), dbLabels, dbData) != 0 || dbData.empty())
            {
                return nullptr;
            }

            auto &entry = gDbCache[dbPath];
            entry.labels = std::move(dbLabels);
            entry.data = std::move(dbData);
            entry.loaded = true;
            if (!timeEc)
            {
                entry.lastWriteTime = nowWriteTime;
            }
            if (!sizeEc)
            {
                entry.fileSize = nowFileSize;
            }
            return &entry;
        }

        return &it->second;
    }
} // namespace

/*
FeatureMatcher::match performs feature matching between a target feature vector and a database of feature vectors
using the specified distance metric. It returns true if a valid match is found and populates bestMatch
*/
bool FeatureMatcher::match(
    const std::vector<float> &targetFeatures,
    const std::string &dbPath,
    MetricType metricType,
    MatchResult &bestMatch)
{
    // Load the feature database from cache or disk
    const CachedFeatureDb *cachedDb = loadCachedDb(dbPath);
    if (cachedDb == nullptr || cachedDb->data.empty())
    {
        std::cout << "[MATCH] DB load failed/empty: " << dbPath << "\n";
        return false;
    }
    const auto &dbLabels = cachedDb->labels;
    const auto &dbData = cachedDb->data;

    // Create the appropriate distance metric object based on the specified metric type
    auto distanceMetric = MetricFactory::create(metricType);
    if (!distanceMetric)
    {
        std::cout << "[MATCH] invalid metric for DB: " << dbPath << "\n";
        return false;
    }

    // Baseline uses scaled Euclidean distance:
    // d(x,y)=sqrt(sum_i ((x_i-y_i)^2 / sigma_i^2)),
    // where sigma_i is std-dev of dimension i over DB.
    std::vector<double> invStd;
    if (metricType == MetricType::SSD)
    {
        const size_t dim = targetFeatures.size();
        if (dim == 0)
        {
            std::cout << "[MATCH] empty target feature vector\n";
            return false;
        }
        invStd.assign(dim, 1.0);

        std::vector<double> mean(dim, 0.0);
        std::vector<double> sqMean(dim, 0.0);
        size_t usedRows = 0;
        for (const auto &row : dbData)
        {
            if (row.size() != dim)
                continue;
            ++usedRows;
            for (size_t i = 0; i < dim; ++i)
            {
                const double v = row[i];
                mean[i] += v;
                sqMean[i] += v * v;
            }
        }
        if (usedRows == 0)
        {
            std::cout << "[MATCH] no compatible DB rows (dimension mismatch)\n";
            return false;
        }
        for (size_t i = 0; i < dim; ++i)
        {
            mean[i] /= static_cast<double>(usedRows);
            sqMean[i] /= static_cast<double>(usedRows);
            const double var = std::max(0.0, sqMean[i] - mean[i] * mean[i]);
            const double sigma = std::sqrt(var);
            // Avoid exploding weights on nearly-constant dimensions.
            invStd[i] = (sigma > 1e-6) ? (1.0 / sigma) : 1.0;
        }
    }
    // Iterate through the database and compute distances to find the best match based on the specified metric
    bool found = false;
    MatchResult best{"", "", 0.0f};
    for (size_t i = 0; i < dbData.size(); ++i)
    {
        float distance = std::numeric_limits<float>::infinity();
        if (metricType == MetricType::SSD)
        {
            // For SSD, compute the scaled Euclidean distance using the precomputed inverse standard deviations
            if (dbData[i].size() != targetFeatures.size())
                continue;
            double acc = 0.0;
            for (size_t k = 0; k < targetFeatures.size(); ++k)
            {
                const double diff = static_cast<double>(targetFeatures[k]) - static_cast<double>(dbData[i][k]);
                const double z = diff * invStd[k];
                acc += z * z;
            }
            distance = static_cast<float>(std::sqrt(acc));
        }
        else
        {
            distance = distanceMetric->compute(targetFeatures, dbData[i]);
        }
        if (!std::isfinite(distance))
        {
            continue;
        }
        if (!found || distance < best.distance)
        {
            // Update the best match if this is the first valid match found or if the distance is smaller than the current best
            found = true;
            best.label = dbLabels[i];
            best.filename = dbLabels[i];
            best.distance = distance;
        }
    }
    if (!found)
    {
        std::cout << "[MATCH] no finite-distance match found\n";
        return false;
    }

    bestMatch = best;
    std::cout << "[MATCH] best label=" << bestMatch.label
              << " dist=" << bestMatch.distance
              << " metric=" << MetricFactory::metricTypeToString(metricType) << "\n";
    return true;
}
