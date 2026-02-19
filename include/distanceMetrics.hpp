/*
Claire Liu, Yu-Jing Wei
distanceMetrics.hpp

Path: include/distanceMetrics.hpp
Description: Header file for distanceMetrics.cpp to
             compute distances between feature vectors.
*/

#pragma once // Include guard

#include "IDistanceMetric.hpp"
#include "metricFactory.hpp"
#include <vector>

/*
    Sum of Squared Distance (SSD) metric
    Computes the sum of squared differences between two feature vectors.
    Lower values indicate more similar features.
 */
struct SumSquaredDistance : public IDistanceMetric
{
    // Constructor to initialize the metric type
    SumSquaredDistance(MetricType mt) : IDistanceMetric(mt) {}
    // Override the compute function to calculate the SSD between two vectors
    float compute(const std::vector<float> &v1,
                  const std::vector<float> &v2)
        const override;
};

/*
    Histogram Intersection metric
    Computes the histogram intersection between two feature vectors(already normalized).
    Higher values indicate more similar features, so we convert it to a distance by subtracting from 1.
 */
struct HistogramIntersection : public IDistanceMetric
{
    // Constructor to initialize the metric type
    HistogramIntersection(MetricType mt) : IDistanceMetric(mt) {}
    // Override the compute function to calculate the histogram intersection distance between two vectors
    float compute(const std::vector<float> &v1,
                  const std::vector<float> &v2)
        const override;
};

/*
    Cosine Distance metric
    Computes the cosine distance between two feature vectors.
    Lower values indicate more similar features.
*/
struct CosDistance : public IDistanceMetric
{
    // Constructor to initialize the metric type
    CosDistance(MetricType mt) : IDistanceMetric(mt) {}
    // Override the compute function to calculate the cosine distance between two vectors
    float compute(const std::vector<float> &v1,
                  const std::vector<float> &v2)
        const override;
};
