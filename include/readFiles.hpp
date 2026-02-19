/*
Claire Liu, Yu-Jing Wei
readFiles.hpp

Path: include/readFiles.hpp
Description: Header file for readFiles.cpp to read image files in a directory.
*/

#pragma once // Include guard

#include <opencv2/opencv.hpp>

/*
ReadFiles class provides static methods to read files from a directory and
to read features from a CSV file.
- readFilesInDir(
        char *dirname,
        std::vector<std::string> &files):
    A static method that takes a directory name and a reference to a vector of strings.
    It reads all the files in the specified directory and stores their full paths in
    the provided vector. It returns an integer status code (e.g., 0 for success, -1 for failure).
- readFeaturesFromCSV(
        char *filename,
        std::vector<char *> &filenames,
        std::vector<std::vector<float>> &data):
    A static method that takes a CSV filename, a reference to a vector of character pointers
    for filenames, and a reference to a vector of vectors of floats for feature data.
    It reads the CSV file, extracts the filenames and their corresponding feature vectors,
    and stores them in the provided vectors.
    It returns an integer status code (e.g., 0 for success, -1 for failure).
- isTargetImageInDatabase(
        const char *targetPath,
        const std::vector<char *> &dbFilenames):
    A static method that checks if a target image (specified by its file path) is present
    in a database of filenames. It takes the target image path and a vector of database filenames
    as input and returns a boolean value indicating whether the target image is found in the database.
*/
class ReadFiles
{
public:
    static int readFilesInDir(
        char *dirname,
        std::vector<std::string> &files);

    static int readFeaturesFromCSV(
        const char *filename,
        std::vector<std::string> &filenames,
        std::vector<std::vector<float>> &data);

    static bool isTargetImageInDatabase(
        const char *targetPath,
        const char *dbFilename);
};
