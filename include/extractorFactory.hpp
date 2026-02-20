/*
Claire Liu, Yu-Jing Wei
extractorFactory.hpp

Path: include/extractorFactory.hpp
Description: Header file for extractorFactory.cpp to
             create feature extractor instances based on extractor type.
*/

#pragma once // Include guard

#include <memory>
#include <vector>

class IExtractor;

/*
Enumeration for different types of feature extractors supported in the project.
*/
enum ExtractorType
{
    BASELINE,
    CNN,
    EIGENSPACE,
    UNKNOWN_EXTRACTOR
};

/*
ExtractorFactory class that provides a static method to create instances of IExtractor
based on the specified ExtractorType.
- create(ExtractorType type): A factory method that takes an ExtractorType and returns a
                    shared pointer to an IExtractor instance corresponding to that type.
                    If the type is unrecognized, it returns nullptr.
- stringToExtractorType(const char *typeStr): A utility method that converts a string
                    representation of an extractor type (e.g., "baseline", "cnn") to
                    the corresponding ExtractorType enum value. If the string does not match
                    any known extractor type, it returns UNKNOWN_EXTRACTOR.
- extractorTypeToString(ExtractorType type): A utility method that converts an ExtractorType enum
                    value back to its string representation for display purposes. If the
                    type is unrecognized, it returns "Unknown".
*/
class ExtractorFactory
{
public:
    static std::shared_ptr<IExtractor> create(ExtractorType type);
    static ExtractorType stringToExtractorType(const char *typeStr);
    static std::string extractorTypeToString(ExtractorType type);
};