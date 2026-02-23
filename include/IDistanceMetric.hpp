/*
Claire Liu, Yu-Jing Wei
IDistanceMetric.hpp

Path: include/IDistanceMetric.hpp
Description: Header file for IDistanceMetric to define the interface for distance metrics.
*/

#pragma once
#include <vector>
#include <string>
#include "metricFactory.hpp"

/*
IDistanceMetric is an abstract base class that defines the interface for distance metrics
used to compare feature vectors.
*/
class IDistanceMetric
{
public:
    // Virtual destructor to ensure proper cleanup of derived classes
    virtual ~IDistanceMetric() = default;

    virtual float compute(const std::vector<float> &features1, const std::vector<float> &features2) const = 0;
    virtual std::string type() { return MetricFactory::metricTypeToString(type_); }

protected:
    // Protected member variable to store the type of the distance metric
    MetricType type_;
    // Constructor to initialize the metric type in derived classes
    explicit IDistanceMetric(MetricType type) : type_(type) {}
};