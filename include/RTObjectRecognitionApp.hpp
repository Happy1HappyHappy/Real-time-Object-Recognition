/*
Claire Liu, Yu-Jing Wei
RTObjectRecognitionApp.hpp

Path: include/RTObjectRecognitionApp.hpp
Description: Declares the RTObjectRecognitionApp class for real-time object recognition.
*/

#pragma once // Include guard

#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "extractorFactory.hpp"
#include "preProcessor.hpp"

/*
AppState struct to hold the state of the application, including flags for different modes,
training state, recording state, and paths for results and data directories.
*/
struct AppState
{
    bool baselineOn = false;
    bool cnnOn = false;
    bool eigenspaceOn = false;
    bool debugOn = false;
    bool showThresholdWindow = false;
    bool showCleanedWindow = false;
    bool showRegionMapWindow = false;

    bool trainingOn = false;
    std::string label;
    DetectionResult lastDetection;
    std::string predExtractor = "none";
    std::string predLabel = "n/a";
    float predDistance = 0.0f;
    bool hasPrediction = false;
    bool hasBaselinePrediction = false;
    bool hasCnnPrediction = false;
    bool hasEigenspacePrediction = false;
    bool rejectUnknown = true;
    std::string unknownLabel = "UNKNOWN";
    std::string baselineLabel = "n/a";
    std::string cnnLabel = "n/a";
    std::string eigenspaceLabel = "n/a";
    float baselineDistance = 0.0f;
    float cnnDistance = 0.0f;
    float eigenspaceDistance = 0.0f;
    float baselineUnknownThreshold = 1.3f;
    float cnnUnknownThreshold = 30.0f;
    float eigenspaceUnknownThreshold = 0.35f;
    std::vector<cv::Rect> predictedBoxes;
    std::vector<std::string> predictedTexts;
    std::vector<std::string> cachedCnnLabels;
    std::vector<float> cachedCnnDistances;

    bool recordingOn = false;
    cv::VideoWriter writer;
    double fps = 24.0;
    int cnnIntervalFrames = 3; // run CNN every N frames
    int maxCnnRegionsPerFrame = 2; // cap CNN inference count per frame

    std::filesystem::path resultsDir = "./results/";
    std::filesystem::path dataDir = "./data/";
};

/*
RTObjectRecognitionApp class to handle real-time object recognition using feature matching.
*/
class RTObjectRecognitionApp
{
public:
    int run();

private:
    static std::string dbPathFor(const AppState &st, ExtractorType type);
    void drawOverlay(cv::Mat &display, const AppState &st);
    void enrollToDb(
        const AppState &st,
        ExtractorType type,
        const cv::Mat &embImage,
        const std::string &savedPath,
        const RegionFeatures *bestRegion = nullptr,
        const cv::Mat *sourceFrame = nullptr);
    std::string sanitizeLabel(std::string s);
    std::string timestampNow();
    void handleTrainingKey(AppState &st, int key, const cv::Mat &frame, const DetectionResult &det);
    bool handleKey(AppState &st, int key, const cv::Size &refS);
};
