/*
Claire Liu, Yu-Jing Wei
preTrainer.cpp

Path: src/offline/preTrainer.cpp
Description: Pre-training utilities for image processing.
*/

#include <cstdio>
#include <cstdlib>
#include "csvUtil.hpp"
#include "extractorFactory.hpp"
#include "extractor.hpp"
#include "preProcessor.hpp"
#include "preTrainerCLI.hpp"
#include "readFiles.hpp"
#include "utilities.hpp"

/*
Helper function to extract features from a list of image paths using the specified extractor
and save them to the output path.
- @param imagePaths A vector of strings containing the paths to the images to be processed.
- @param extractor A shared pointer to an IExtractor instance used for feature extraction.
- @param outPath The path to the output CSV file where the extracted features will be saved.
- @return 0 on success, non-zero value on error.
*/
int extractFeaturesToFile(
    const std::vector<std::string> &imagePaths,
    const std::shared_ptr<IExtractor> &extractor,
    ExtractorType extractorType,
    const std::string &outPath)
{
    std::vector<float> featureVector; // vector to hold features for each image
    // extract features for each image
    for (const auto &path : imagePaths)
    {
        featureVector.clear(); // clear the feature vector for each image
        cv::Mat img = cv::imread(path);
        if (img.empty())
        {
            printf("Warning: cannot read image %s\n", path.c_str());
            continue;
        }

        // Pre-train mode: always use only the best detected region
        DetectionResult det = PreProcessor::detect(img, /*keepAllRegions*/ false);
        if (!det.valid || det.embImage.empty())
        {
            printf("Warning: no valid region in %s\n", path.c_str());
            continue;
        }

        int rc = -1;
        if (extractorType == ExtractorType::BASELINE)
        {
            // For the baseline extractor, we extract features directly from the best detected region
            rc = extractor->extractRegion(det.bestRegion, &featureVector);
        }
        else
        {
            if (extractorType == ExtractorType::CNN)
            {
                // For CNN, we need to prepare the embedding image by rotating and resizing the detected region
                cv::Mat cnnInput;
                const bool prepOk = utilities::prepEmbeddingImage(img, det.bestRegion, cnnInput, 224, true);
                if (!prepOk || cnnInput.empty())
                {
                    printf("Warning: CNN prep failed for %s\n", path.c_str());
                    continue;
                }
                rc = extractor->extractMat(cnnInput, &featureVector);
            }
            else
            {
                rc = extractor->extractMat(det.embImage, &featureVector);
            }
        }
        if (rc != 0)
        {
            printf("Warning: extract failed for %s\n", path.c_str());
            continue;
        }

        // save features in an image to output file
        csvUtil::append_image_data_csv(outPath.c_str(), path.c_str(), featureVector, 0);
    }
    printf("Done. Processed %lu images.\n", imagePaths.size());
    return 0; // Success
}

int main(int argc, char *argv[])
{
    // get the directory path, extractor type and output file path
    std::string dirname;
    std::string outputBase;
    std::string modelPath;
    ExtractorType extractorType;
    const int parseRc = PreTrainerCLI::parseCLI(argc, argv, dirname, extractorType, outputBase, &modelPath);
    if (parseRc != 0)
    {
        return (parseRc > 0) ? 0 : -1;
    }

    if (!modelPath.empty())
    {
        setenv("RTOR_CNN_MODEL", modelPath.c_str(), 1);
    }

    // read the files in the directory, get the file paths
    std::vector<std::string> imagePaths;
    ReadFiles::readFilesInDir((char *)dirname.c_str(), imagePaths);

    // generate the output file path
    std::string outPath = csvUtil::setOutputFilename(outputBase, extractorType);

    // Ensure the output feature CSV file is empty
    csvUtil::clearExistingFile(outPath.c_str());

    // create the extractor based on the specified type
    std::shared_ptr<IExtractor> extractor;
    try
    {
        extractor = ExtractorFactory::create(extractorType);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Extractor creation failed: " << e.what() << std::endl;
        return -1;
    }
    // extract features for each image and save to output file
    extractFeaturesToFile(imagePaths, extractor, extractorType, outPath);

    return 0;
}
