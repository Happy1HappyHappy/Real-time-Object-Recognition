/*
Claire Liu, Yu-Jing Wei
main.cpp

Path: src/online/main.cpp
Description: Entry point for the real-time object recognition application.
*/
#include "RTObjectRecognitionApp.hpp"

int main()
{
    try
    {
        RTObjectRecognitionApp app;
        app.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return -1;
    }
}