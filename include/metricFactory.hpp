/*
Claire Liu, Yu-Jing Wei
metricFactory.hpp

Path: include/metricFactory.hpp
Description: Header file for metricFactory.cpp to create feature metric
instances based on feature type.
*/

#pragma once // Include guard

#include <memory>
#include <vector>

// Forward declaration of IDistanceMetric to avoid circular dependency
class IDistanceMetric;

/*
Enumeration for different distance metric types that can be used to compare feature vectors.
- SSD: Sum of Squared Distance, which computes the sum of squared differences
    between two feature vectors. Lower values indicate more similar features.
- HIST_INTERSECTION: Histogram Intersection, which computes the histogram
    intersection between two feature vectors (already normalized).
    Higher values indicate more similar features, so we convert it to a distance
    by subtracting from 1.
- COSINE: Cosine similarity, which computes the cosine of the angle between two feature vectors.
    Higher values indicate more similar features, so we convert it to a distance
    by subtracting from 1.
- UNKNOWN_METRIC: A default value for unrecognized metric types.
*/
enum MetricType
{
    SSD,
    HIST_INTERSECTION,
    COSINE,
    UNKNOWN_METRIC
};

/*
MetricFactory class that provides a static method to create instances of IDistanceMetric
based on the specified MetricType.
- create(MetricType type): A factory method that takes a MetricType and returns a
                    shared pointer to an IDistanceMetric instance corresponding to that type.
                    If the type is unrecognized, it returns nullptr.
- metricTypeToString(MetricType type): A utility method that converts a MetricType enum
                    value back to its string representation for display purposes. If the
                    type is unrecognized, it returns "Unknown".
*/
class MetricFactory
{
public:
    static std::shared_ptr<IDistanceMetric> create(MetricType type);
    static std::string metricTypeToString(MetricType type);
};
