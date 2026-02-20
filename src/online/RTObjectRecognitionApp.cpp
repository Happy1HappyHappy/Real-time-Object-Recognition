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
#include "RTObjectRecognitionApp.hpp"

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

    for (;;)
    {
        capdev >> frame;
        if (frame.empty())
            break;

        cv::Mat currentFrame = frame;

        // TODO:
        // Pre-process the frame and get the region of interest (ROI)

        // apply feature extractors if they are on
        if (st.baselineOn)
        {
            // currentFrame = applyBaseline(currentFrame);
        }
        if (st.cnnOn)
        {
            // currentFrame = applyCNN(currentFrame);
        }
        if (st.eigenspaceOn)
        {
            // currentFrame = applyEigenspace(currentFrame);
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
            handleTrainingKey(st, key, frame);
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

    if (st.recordingOn)
    {
        cv::circle(display, {30, 120}, 10, {0, 0, 255}, -1);
        cv::putText(display, "REC", {50, 132}, cv::FONT_HERSHEY_DUPLEX, 0.8, {0, 0, 255}, 2);
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

void RTObjectRecognitionApp::handleTrainingKey(AppState &st, int key, const cv::Mat &frame)
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
        else
        {
            std::string safe = sanitizeLabel(st.label);
            std::string out = (st.dataDir / (safe + "_" + timestampNow() + ".png")).string();
            if (cv::imwrite(out, frame))
                std::cout << "[TRAIN] Saved " << out << "\n";
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
