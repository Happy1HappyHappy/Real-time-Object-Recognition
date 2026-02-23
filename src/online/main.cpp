/*
Claire Liu, Yu-Jing Wei
main.cpp

Path: src/online/main.cpp
Description: Entry point for the real-time object recognition application.
*/
#include "RTObjectRecognitionApp.hpp"

/*
main function that initializes and runs the RTObjectRecognitionApp.
It also catches any exceptions thrown during execution and prints an
error message before exiting with a non-zero status code.
*/
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