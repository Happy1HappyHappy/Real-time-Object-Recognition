/*
  Claire Liu, Yu-Jing Wei
  distanceMetrics.cpp

  Path: src/utils/distanceMetrics.cpp
  Description: Implements distance metrics for feature vectors.
*/

#include "distanceMetrics.hpp"
#include "csvUtil.hpp"
#include "opencv2/opencv.hpp"
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
#include <numeric>

/*
Sum of Squared Distance (SSD) metric
Computes the sum of squared differences between two feature vectors.
Lower values indicate more similar features.

- @param v1 The first feature vector.
- @param v2 The second feature vector.
- @return The computed SSD distance between the two vectors.
    If the vectors are of different lengths, it returns INFINITY.
*/
float SumSquaredDistance::compute(
    const std::vector<float> &v1,
    const std::vector<float> &v2) const
{
    // v1 and v2 not the same size, return infinity to indicate they cannot be compared
    if (v1.size() != v2.size())
    {
        printf("Feature vectors size does not match\n");
        return std::numeric_limits<float>::infinity();
    }

    float sum = 0.0f; // Initialize the sum of squared differences as 0
    for (size_t i = 0; i < v1.size(); ++i)
    {
        float diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sum;
}

/*
Histogram Intersection metric
Computes the rghistogram intersection between two feature vectors (already normalized).
Higher values indicate more similar features, so we convert it to a distance by
subtracting from 1.

- @param v1 The first feature vector (normalized).
- @param v2 The second feature vector (normalized).
- @return The computed histogram intersection distance between the two vectors.
    If the vectors are of different lengths, it returns INFINITY.
*/
float HistogramIntersection::compute(
    const std::vector<float> &v1,
    const std::vector<float> &v2) const
{
    // v1 and v2 not the same size, return infinity to indicate they cannot be compared
    if (v1.size() != v2.size())
    {
        printf("Feature vectors size does not match\n");
        return std::numeric_limits<float>::infinity();
    }

    float intersection = 0.0f; // Initialize the intersection value as 0
    for (size_t i = 0; i < v1.size(); ++i)
    {
        intersection += std::min(v1[i], v2[i]);
    }
    return 1.0f - intersection; // Convert similarity to distance
}

/*
 * Cosine Distance Metric
 *
 * Computes the cosine distance between two feature vectors.
 *
 * @param v1 The first feature vector.
 * @param v2 The second feature vector.
 * @return The computed cosine distance.
 * Returns 1.0 if either vector has zero magnitude (undefined angle).
 */
float CosDistance::compute(
    const std::vector<float> &v1,
    const std::vector<float> &v2) const
{
    // Inner Product
    double dot = std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0);

    // sum square
    double sum_sq1 = std::inner_product(v1.begin(), v1.end(), v1.begin(), 0.0);
    double sum_sq2 = std::inner_product(v2.begin(), v2.end(), v2.begin(), 0.0);

    // L2 norm
    double norm1 = std::sqrt(sum_sq1);
    double norm2 = std::sqrt(sum_sq2);

    // avoid 0
    if (norm1 == 0.0 || norm2 == 0.0)
    {
        return 1.0f;
    }

    // Cosine Similarity
    double similarity = dot / (norm1 * norm2);

    // Transfer Cosine distance
    return static_cast<float>(1.0 - similarity);
}