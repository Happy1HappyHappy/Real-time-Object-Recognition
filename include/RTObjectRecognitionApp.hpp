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

    bool trainingOn = false;
    std::string label;
    DetectionResult lastDetection;
    std::string predExtractor = "none";
    std::string predLabel = "n/a";
    float predDistance = 0.0f;
    bool hasPrediction = false;

    bool recordingOn = false;
    cv::VideoWriter writer;
    double fps = 24.0;

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
    void enrollToDb(const AppState &st, ExtractorType type, const cv::Mat &embImage, const std::string &savedPath);
    std::string sanitizeLabel(std::string s);
    std::string timestampNow();
    void handleTrainingKey(AppState &st, int key, const cv::Mat &frame, const DetectionResult &det);
    bool handleKey(AppState &st, int key, const cv::Size &refS);
};
