/*
  Claire Liu, Yu-Jing Wei
  extractorFactory.cpp

  Path: src/utils/extractorFactory.cpp
  Description: Implements the factory method for creating feature extractor instances.
*/

#include "extractorFactory.hpp"
#include "extractor.hpp"
#include <memory>
#include <unordered_map>

/*
ExtractorFactory::create(ExtractorType type)
This static method creates and returns a shared pointer to an IExtractor instance based
on the specified ExtractorType. It uses a switch statement to determine which type of
extractor to create:
- BASELINE, it creates and returns a shared pointer to a BaselineExtractor instance.
- CNN, it creates and returns a shared pointer to a CNNExtractor instance.
 - UNKNOWN_EXTRACTOR or any unrecognized type, it returns nullptr to indicate that no valid
    extractor could be created.
*/
std::shared_ptr<IExtractor> ExtractorFactory::create(ExtractorType type)
{
    switch (type)
    {
    case BASELINE:
        return std::make_shared<BaselineExtractor>(type);
    case CNN:
        return std::make_shared<CNNExtractor>(type);
    default:
        throw std::invalid_argument("Unknown ExtractorType");
    }
}

/*
ExtractorFactory::stringToExtractorType(const char *typeStr)
This static method converts a string representation of an extractor type to the corresponding
ExtractorType enum value. It compares the input string to known extractor type strings:
- "baseline" returns BASELINE
- "cnn" returns CNN
If the input string does not match any known extractor type, it returns UNKNOWN_EXTRACTOR.
*/
ExtractorType ExtractorFactory::stringToExtractorType(const char *typeStr)
{
    static const std::unordered_map<std::string, ExtractorType> typeMap = {
        {"baseline", BASELINE},
        {"cnn", CNN}};

    auto it = typeMap.find(typeStr);
    return (it != typeMap.end()) ? it->second : UNKNOWN_EXTRACTOR;
}

/*
ExtractorFactory::extractorTypeToString(ExtractorType type)
This static method converts an ExtractorType enum value back to its string representation for
display purposes. It uses a switch statement to return the corresponding string for each
ExtractorType:
- BASELINE returns "baseline"
- CNN returns "cnn"
If the type is unrecognized, it returns "Unknown".
*/
std::string ExtractorFactory::extractorTypeToString(ExtractorType type)
{
    static const std::unordered_map<ExtractorType, std::string> reverseMap = {
        {BASELINE, "baseline"},
        {CNN, "cnn"}};

    auto it = reverseMap.find(type);
    return (it != reverseMap.end()) ? it->second : "Unknown";
}
