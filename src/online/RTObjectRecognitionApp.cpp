/*
Claire Liu, Yu-Jing Wei
RTObjectRecognitionApp.cpp

Path: src/online/RTObjectRecognitionApp.cpp
Description: Real-time object recognition using feature matching.
*/

#include <iostream>   // cout, cerr
#include <string>     // std::string
#include <ctime>      // std::time, std::localtime
#include <iomanip>    // std::put_time
#include <sstream>    // std::ostringstream
#include <algorithm>  // std::remove_if
#include <cctype>     // std::isalnum
#include <filesystem> // std::filesystem
#include <opencv2/opencv.hpp>
#include "csvUtil.hpp"
#include "extractorFactory.hpp"
#include "featureMatcher.hpp"
#include "IExtractor.hpp"
#include "preProcessor.hpp"
#include "RTObjectRecognitionApp.hpp"

std::string RTObjectRecognitionApp::dbPathFor(const AppState &st, ExtractorType type)
{
    switch (type)
    {
    case BASELINE:
        return (st.dataDir / "features_baseline.csv").string();
    case CNN:
        return (st.dataDir / "features_cnn.csv").string();
    case EIGENSPACE:
        return (st.dataDir / "features_eigenspace.csv").string();
    default:
        return "";
    }
}

void RTObjectRecognitionApp::enrollToDb(
    const AppState &st,
    ExtractorType type,
    const cv::Mat &embImage,
    const std::string &savedPath)
{
    auto extractor = ExtractorFactory::create(type);
    std::vector<float> featureVector;
    if (extractor->extractMat(embImage, &featureVector) != 0 || featureVector.empty())
    {
        std::cerr << "[TRAIN] feature extraction failed for " << ExtractorFactory::extractorTypeToString(type) << "\n";
        return;
    }

    const std::string dbPath = dbPathFor(st, type);
    if (dbPath.empty())
        return;
    csvUtil::append_image_data_csv(dbPath.c_str(), savedPath.c_str(), featureVector, 0);
    std::cout << "[TRAIN] appended " << ExtractorFactory::extractorTypeToString(type) << " features to " << dbPath << "\n";
}
/*
RTObjectRecognitionApp class to handle real-time object recognition using feature matching.
*/
int RTObjectRecognitionApp::run()
{
    AppState st;

    // Ensure results and data directories exist
    std::error_code ec;
    std::filesystem::create_directories(st.resultsDir, ec);
    std::filesystem::create_directories(st.dataDir, ec);

    cv::namedWindow("Video", 1);

    cv::Mat frame;

    // open the default video camera
    cv::VideoCapture capdev(0);
    if (!capdev.isOpened())
    {
        std::cerr << "Unable to open video device\n";
        return -1;
    }

    // get camera frame size
    cv::Size refS((int)capdev.get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Expected size: " << refS.width << " " << refS.height << "\n";

    // create the extractor based on the specified type
    auto baselineExtractor = ExtractorFactory::create(ExtractorType::BASELINE);
    auto cnnExtractor = ExtractorFactory::create(ExtractorType::CNN);
    auto eigenspaceExtractor = ExtractorFactory::create(ExtractorType::EIGENSPACE);
    const std::string baselineDbPath = dbPathFor(st, BASELINE);
    const std::string cnnDbPath = dbPathFor(st, CNN);
    const std::string eigenspaceDbPath = dbPathFor(st, EIGENSPACE);
    size_t frameId = 0;

    for (;;)
    {
        capdev >> frame;
        if (frame.empty())
            break;
        ++frameId;
        std::cout << "[FRAME " << frameId << "] captured\n";

        cv::Mat currentFrame = frame.clone();
        st.lastDetection = PreProcessor::detect(currentFrame);
        currentFrame = st.lastDetection.debugFrame.clone();
        if (st.lastDetection.valid)
        {
            std::cout << "[DETECT] valid bbox=("
                      << st.lastDetection.bestBBox.x << ","
                      << st.lastDetection.bestBBox.y << ","
                      << st.lastDetection.bestBBox.width << ","
                      << st.lastDetection.bestBBox.height << ")\n";
        }
        else
        {
            std::cout << "[DETECT] no valid region\n";
        }

        st.hasPrediction = false;
        st.predExtractor = "none";
        st.predLabel = "n/a";
        st.predDistance = 0.0f;

        std::vector<float> featureVector;
        MatchResult matchResult;
        if (st.baselineOn && st.lastDetection.valid)
        {
            std::cout << "[CLASSIFY] mode=baseline\n";
            featureVector.clear();
            const bool extractOk = (baselineExtractor->extractMat(st.lastDetection.embImage, &featureVector) == 0);
            if (!extractOk)
            {
                std::cerr << "Baseline extractor failed on current frame.\n";
            }
            std::cout << "[FEATURE] dim=" << featureVector.size() << " db=" << baselineDbPath << "\n";
            if (extractOk && FeatureMatcher::match(featureVector, baselineDbPath, MetricType::SSD, matchResult))
            {
                st.hasPrediction = true;
                st.predExtractor = "baseline";
                st.predLabel = matchResult.label;
                st.predDistance = matchResult.distance;
                std::cout << "[PRED] " << st.predLabel << " dist=" << st.predDistance << "\n";
            }
            else
            {
                std::cout << "[PRED] no match (baseline)\n";
            }
        }
        else if (st.cnnOn && st.lastDetection.valid)
        {
            std::cout << "[CLASSIFY] mode=cnn\n";
            featureVector.clear();
            const bool extractOk = (cnnExtractor->extractMat(st.lastDetection.embImage, &featureVector) == 0);
            if (!extractOk)
            {
                std::cerr << "CNN extractor failed on current frame.\n";
            }
            std::cout << "[FEATURE] dim=" << featureVector.size() << " db=" << cnnDbPath << "\n";
            if (extractOk && FeatureMatcher::match(featureVector, cnnDbPath, MetricType::COSINE, matchResult))
            {
                st.hasPrediction = true;
                st.predExtractor = "cnn";
                st.predLabel = matchResult.label;
                st.predDistance = matchResult.distance;
                std::cout << "[PRED] " << st.predLabel << " dist=" << st.predDistance << "\n";
            }
            else
            {
                std::cout << "[PRED] no match (cnn)\n";
            }
        }
        else if (st.eigenspaceOn && st.lastDetection.valid)
        {
            std::cout << "[CLASSIFY] mode=eigenspace\n";
            featureVector.clear();
            const bool extractOk = (eigenspaceExtractor->extractMat(st.lastDetection.embImage, &featureVector) == 0);
            if (!extractOk)
            {
                std::cerr << "Eigenspace extractor failed on current frame.\n";
            }
            std::cout << "[FEATURE] dim=" << featureVector.size() << " db=" << eigenspaceDbPath << "\n";
            if (extractOk && FeatureMatcher::match(featureVector, eigenspaceDbPath, MetricType::COSINE, matchResult))
            {
                st.hasPrediction = true;
                st.predExtractor = "eigenspace";
                st.predLabel = matchResult.label;
                st.predDistance = matchResult.distance;
                std::cout << "[PRED] " << st.predLabel << " dist=" << st.predDistance << "\n";
            }
            else
            {
                std::cout << "[PRED] no match (eigenspace)\n";
            }
        }
        else if (st.baselineOn || st.cnnOn || st.eigenspaceOn)
        {
            std::cout << "[CLASSIFY] skipped (no valid detection)\n";
        }
        else
        {
            std::cout << "[CLASSIFY] skipped (no mode enabled)\n";
        }

        // if recordingOn, write the current frame to the video file
        if (st.recordingOn && st.writer.isOpened())
            st.writer.write(currentFrame);

        // use a clone of the current frame for display
        // so we can draw overlays without affecting the original frame
        cv::Mat display = currentFrame.clone();
        drawOverlay(display, st);
        cv::imshow("Video", display);

        int key = cv::waitKey(1);

        // trainingOn mode key handling
        if (st.trainingOn)
        {
            // trainingOn mode key handling：label input, save, cancel
            handleTrainingKey(st, key, frame, st.lastDetection);
            continue;
        }

        // general key handling
        if (!handleKey(st, key, refS))
            break;

        // screenshot
        if (key == 's' || key == 'S')
        {
            std::string path = (st.resultsDir / ("screenshot_" + timestampNow() + ".png")).string();
            if (cv::imwrite(path, currentFrame))
                std::cout << "Saved screenshot: " << path << "\n";
            else
                std::cout << "ERROR: Could not save screenshot.\n";
        }
    }

    // release video writer if still open
    if (st.writer.isOpened())
        st.writer.release();

    return 0;
}

// Helper function to convert boolean to "ON"/"OFF" string
static std::string onOff(bool v) { return v ? "ON" : "OFF"; }

void RTObjectRecognitionApp::drawOverlay(cv::Mat &display, const AppState &st)
{
    if (st.trainingOn)
    {
        cv::putText(display, "TRAINING MODE: type label, ENTER to save, ESC to cancel",
                    {20, 30}, cv::FONT_HERSHEY_DUPLEX, 0.7, {255, 255, 255}, 2, cv::LINE_AA);
        cv::putText(display, "Label: " + st.label,
                    {20, 60}, cv::FONT_HERSHEY_DUPLEX, 0.7, {255, 255, 255}, 2, cv::LINE_AA);
    }
    else
    {
        cv::putText(display, "Press 't' to label+save frame, 'r' record, 's' screenshot, 'q' quit",
                    {20, 30}, cv::FONT_HERSHEY_DUPLEX, 0.7, {255, 255, 255}, 2, cv::LINE_AA);
    }

    std::string status =
        "B(Baseline): " + onOff(st.baselineOn) +
        "   C(CNN): " + onOff(st.cnnOn) +
        "   E(Eigenspace): " + onOff(st.eigenspaceOn);

    cv::putText(display, status,
                {20, 80}, cv::FONT_HERSHEY_DUPLEX, 0.7,
                {255, 255, 255}, 2, cv::LINE_AA);

    cv::putText(display,
                std::string("Detection: ") + (st.lastDetection.valid ? "VALID" : "NONE"),
                {20, 110},
                cv::FONT_HERSHEY_DUPLEX,
                0.65,
                st.lastDetection.valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 180, 255),
                2,
                cv::LINE_AA);

    if (st.hasPrediction)
    {
        std::ostringstream oss;
        oss << "Label: " << st.predLabel
            << "  dist=" << std::fixed << std::setprecision(4) << st.predDistance;
        cv::putText(display, oss.str(),
                    {20, 140}, cv::FONT_HERSHEY_DUPLEX, 0.6, {0, 255, 255}, 2, cv::LINE_AA);

        if (st.lastDetection.valid)
        {
            const cv::Rect box = st.lastDetection.bestBBox & cv::Rect(0, 0, display.cols, display.rows);
            if (box.width > 0 && box.height > 0)
            {
                const std::string boxText = "Predicted Classification: " + st.predLabel;
                int baseline = 0;
                const cv::Size textSize = cv::getTextSize(boxText, cv::FONT_HERSHEY_SIMPLEX, 0.55, 2, &baseline);
                int tx = box.x;
                int ty = box.y - 8;
                if (ty - textSize.height < 0)
                {
                    ty = box.y + textSize.height + 8;
                }
                tx = std::max(0, std::min(tx, display.cols - textSize.width - 6));

                const cv::Rect bg(tx - 3, ty - textSize.height - 3,
                                  textSize.width + 6, textSize.height + baseline + 6);
                cv::rectangle(display, bg, cv::Scalar(0, 0, 0), cv::FILLED);
                cv::putText(display, boxText, cv::Point(tx, ty),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            }
        }
    }
    else if (st.lastDetection.valid && (st.baselineOn || st.cnnOn || st.eigenspaceOn))
    {
        const cv::Rect box = st.lastDetection.bestBBox & cv::Rect(0, 0, display.cols, display.rows);
        if (box.width > 0 && box.height > 0)
        {
            const std::string boxText = "NO MATCH";
            int baseline = 0;
            const cv::Size textSize = cv::getTextSize(boxText, cv::FONT_HERSHEY_SIMPLEX, 0.55, 2, &baseline);
            int tx = box.x;
            int ty = box.y - 8;
            if (ty - textSize.height < 0)
            {
                ty = box.y + textSize.height + 8;
            }
            tx = std::max(0, std::min(tx, display.cols - textSize.width - 6));
            const cv::Rect bg(tx - 3, ty - textSize.height - 3,
                              textSize.width + 6, textSize.height + baseline + 6);
            cv::rectangle(display, bg, cv::Scalar(0, 0, 0), cv::FILLED);
            cv::putText(display, boxText, cv::Point(tx, ty),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 180, 255), 2, cv::LINE_AA);
        }
    }

    if (st.recordingOn)
    {
        cv::circle(display, {30, 170}, 10, {0, 0, 255}, -1);
        cv::putText(display, "REC", {50, 182}, cv::FONT_HERSHEY_DUPLEX, 0.8, {0, 0, 255}, 2);
    }
}

std::string RTObjectRecognitionApp::sanitizeLabel(std::string s)
{
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c)
                           { return !(std::isalnum(c) || c == '_' || c == '-'); }),
            s.end());
    return s;
}

std::string RTObjectRecognitionApp::timestampNow()
{
    auto now = std::time(nullptr);
    std::tm *tm_ptr = std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(tm_ptr, "%Y%m%d_%H%M%S");
    return oss.str();
}

void RTObjectRecognitionApp::handleTrainingKey(AppState &st, int key, const cv::Mat &frame, const DetectionResult &det)
{
    if (key == 27)
    { // ESC
        st.trainingOn = false;
        st.label.clear();
        std::cout << "[TRAIN] cancelled\n";
        return;
    }

    if (key == 13 || key == 10)
    { // ENTER
        if (st.label.empty())
        {
            std::cout << "[TRAIN] empty label, not saved\n";
        }
        else if (!det.valid || det.embImage.empty())
        {
            std::cout << "[TRAIN] no valid detection; sample not enrolled\n";
        }
        else
        {
            std::string safe = sanitizeLabel(st.label);
            std::string out = (st.dataDir / (safe + "_" + timestampNow() + ".png")).string();
            if (cv::imwrite(out, det.embImage))
            {
                std::cout << "[TRAIN] Saved " << out << "\n";
                bool anyModeEnabled = st.baselineOn || st.cnnOn || st.eigenspaceOn;
                if (!anyModeEnabled)
                {
                    enrollToDb(st, BASELINE, det.embImage, out);
                }
                if (st.baselineOn)
                {
                    enrollToDb(st, BASELINE, det.embImage, out);
                }
                if (st.cnnOn)
                {
                    enrollToDb(st, CNN, det.embImage, out);
                }
                if (st.eigenspaceOn)
                {
                    enrollToDb(st, EIGENSPACE, det.embImage, out);
                }
            }
            else
                std::cout << "[TRAIN] Failed to save " << out << "\n";
        }
        st.trainingOn = false;
        st.label.clear();
        return;
    }

    if (key == 8 || key == 255)
    { // backspace
        if (!st.label.empty())
            st.label.pop_back();
        return;
    }

    if (key >= 32 && key <= 126)
    { // printable
        st.label.push_back(static_cast<char>(key));
    }
}

bool RTObjectRecognitionApp::handleKey(AppState &st, int key, const cv::Size &refS)
{
    if (key == 'q' || key == 'Q')
        return false;

    if (!st.trainingOn)
    {
        if (key == 'b' || key == 'B')
        {
            st.baselineOn = !st.baselineOn;
            std::cout << "Baseline: " << (st.baselineOn ? "ON" : "OFF") << "\n";
            return true;
        }
        if (!st.trainingOn && (key == 'c' || key == 'C'))
        {
            st.cnnOn = !st.cnnOn;
            std::cout << "CNN: " << (st.cnnOn ? "ON" : "OFF") << "\n";
            return true;
        }

        if (!st.trainingOn && (key == 'e' || key == 'E'))
        {
            st.eigenspaceOn = !st.eigenspaceOn;
            std::cout << "Eigenspace: " << (st.eigenspaceOn ? "ON" : "OFF") << "\n";
            return true;
        }
        if ((key == 't' || key == 'T'))
        {
            st.trainingOn = true;
            st.label.clear();
            std::cout << "[TRAIN] type label (in OpenCV window). ENTER=save, ESC=cancel\n";
            return true;
        }
        if (key == 'r' || key == 'R')
        {
            st.recordingOn = !st.recordingOn;
            if (st.recordingOn)
            {
                std::string path = (st.resultsDir / ("record_" + timestampNow() + ".avi")).string();
                int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
                st.writer.open(path, fourcc, st.fps, refS, true);
                if (!st.writer.isOpened())
                {
                    std::cerr << "ERROR: Failed to open video file for writing.\n";
                    st.recordingOn = false;
                }
                else
                {
                    std::cout << "STARTED Recording: " << path << "\n";
                }
            }
            else
            {
                st.writer.release();
                std::cout << "STOPPED Recording. File saved.\n";
            }
            return true;
        }
    }

    return true;
}
