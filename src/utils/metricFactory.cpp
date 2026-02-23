/*
  Claire Liu, Yu-Jing Wei
  metricFactory.cpp

  Path: src/utils/metricFactory.cpp
  Description: Implements the factory method for creating feature metric instances.
*/

#include "metricFactory.hpp"
#include "distanceMetrics.hpp"
#include <memory>
#include <unordered_map>

/*
MetricFactory::create(MetricType type)
This static method creates and returns a shared pointer to an IDistanceMetric instance based
on the specified MetricType. It uses a switch statement to determine which type of
metric to create:
- SSD, it creates and returns a shared pointer to a SumSquaredDistance instance.
- HIST_INTERSECTION, it creates and returns a shared pointer to a HistogramIntersection instance.
- COSINE, it creates and returns a shared pointer to a CosDistance instance.
- UNKNOWN_METRIC or any unrecognized type, it returns nullptr to indicate that no valid
    metric could be created.
*/
std::shared_ptr<IDistanceMetric> MetricFactory::create(MetricType type)
{
    switch (type)
    {
    case SSD:
        return std::make_shared<SumSquaredDistance>(type);
    case HIST_INTERSECTION:
        return std::make_shared<HistogramIntersection>(type);
    case COSINE:
        return std::make_shared<CosDistance>(type);
    default:
        return nullptr;
    }
}

/*
MetricFactory::metricTypeToString(MetricType type)
This static method converts a MetricType enum value back to its string representation for
display purposes. It uses a switch statement to return the corresponding string for each
metric type:
- SSD returns "ssd"
- HIST_INTERSECTION returns "hist_intersection"
- COSINE returns "cosine"
If the type is unrecognized, it returns "Unknown".
*/
std::string MetricFactory::metricTypeToString(MetricType type)
{
    static const std::unordered_map<MetricType, std::string> typeMap = {
        {SSD, "ssd"},
        {HIST_INTERSECTION, "hist_ix"},
        {COSINE, "cosine"}};

    auto it = typeMap.find(type);
    return (it != typeMap.end()) ? it->second : "Unknown";
}
