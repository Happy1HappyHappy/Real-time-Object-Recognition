/*
  Claire Liu, Yu-Jing Wei
  readFiles.cpp

  Path: src/utils/readFiles.cpp
  Description: Reads image files in a directory.
*/

#include "readFiles.hpp"
#include "csvUtil.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <filesystem>

/*
  Given a directory on the command line, scans through the directory for image files.
  Prints out the full path name for each file.  This can be used as an argument to fopen
  or to cv::imread.

  - @param dirname The path to the directory to scan for image files.
  - @param files A reference to a vector of strings where the full path names of the
                 image files will be stored.
  - @return 0 on success, non-zero value on error.
 */
int ReadFiles::readFilesInDir(char *dirname, std::vector<std::string> &files)
{
    char buffer[256]; // buffer to hold full path names
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    int i;

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL)
    {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }
    printf("Processing directory %s\n", dirname);

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL)
    {

        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif"))
        {

            // printf("processing image file: %s\n", dp->d_name);

            // build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            // printf("full path name: %s\n", buffer);

            // store the full path name in the vector
            files.push_back(std::string(buffer));
        }
    }

    return 0;
}

/*
Reads image features from a CSV file. The CSV file is expected to have a string as the first
column (the filename) and floating point numbers as the remaining columns (the features).
The function populates the provided vectors with the filenames and their corresponding feature data.

- @param filename The path to the CSV file to read.
- @param filenames A reference to a vector of character pointers where the filenames will be stored.
- @param data A reference to a 2D vector of floats where the feature data will be stored. Each inner vector corresponds to the features of one image.
- @return 0 on success, non-zero value on error.
*/
int ReadFiles::readFeaturesFromCSV(const char *filename, std::vector<std::string> &filenames, std::vector<std::vector<float>> &data)
{
    if (csvUtil::read_image_data_csv(filename, filenames, data, 0) != 0)
    {
        printf("Error reading CSV file.\n");
        return -1;
    }
    return 0;
}

/*
Checks if the target image is present in the database by comparing filenames.

- @param targetPath The path to the target image.
- @param dbFilename The path to the database image.
- @return true if the target image is in the database, false otherwise.
*/
bool ReadFiles::isTargetImageInDatabase(const char *targetPath, const char *dbFilename)
{
    namespace fs = std::filesystem;
    if (fs::path(dbFilename).filename() == fs::path(targetPath).filename())
        return true;

    return false;
}