/*
Bruce A. Maxwell

CS 5330 Computer Vision
Spring 2021

CPP functions for reading CSV files with a specific format
- first column is a string containing a filename or path
- every other column is a number

The function returns a std::vector of char* for the filenames and a 2D std::vector of floats for the data
*/

#include <cstdio>
#include <cstring>
#include <vector>
#include "opencv2/opencv.hpp"
#include "csvUtil.hpp"


/*
  Given a filename, and image filename, and the image features, by
  default the function will append a line of data to the CSV format
  file.  If reset_file is true, then it will open the file in 'write'
  mode and clear the existing contents.

  The image filename is written to the first position in the row of
  data. The values in image_data are all written to the file as
  floats.

  The function returns a non-zero value in case of an error.
 */
int csvUtil::append_image_data_csv(const char *filename, const char *image_filename, std::vector<float> &image_data, int reset_file)
{
  char buffer[256];
  char mode[8];
  FILE *fp;

  strcpy(mode, "a");

  if (reset_file)
  {
    strcpy(mode, "w");
  }

  fp = fopen(filename, mode);
  if (!fp)
  {
    printf("Unable to open output file %s\n", filename);
    exit(-1);
  }

  // get the label from the image filename and write it to the first column of the CSV file
  std::string label = csvUtil::getLabel(image_filename);
  std::fwrite(label.c_str(), sizeof(char), label.size(), fp);
  std::fwrite(",", sizeof(char), 1, fp);

  // write the filename and the feature vector to the CSV file
  strcpy(buffer, image_filename);
  std::fwrite(buffer, sizeof(char), strlen(buffer), fp);
  for (int i = 0; i < image_data.size(); i++)
  {
    char tmp[256];
    snprintf(tmp, sizeof(tmp), ",%.4f", image_data[i]);
    std::fwrite(tmp, sizeof(char), strlen(tmp), fp);
  }

  std::fwrite("\n", sizeof(char), 1, fp); // EOL

  fclose(fp);

  return (0);
}

/*
clears the contents of the specified file.
@param filename The path to the file to be cleared.
@return 0 on success, non-zero value on error.
*/
int csvUtil::clearExistingFile(const char *filename)
{
  FILE *f = fopen(filename, "w");
  if (f)
    fclose(f);
  return 0;
}

/*
checks if the specified file exists.
@param filename The path to the file to be checked.
@return 1 if the file exists, 0 otherwise.
*/
int csvUtil::fileExists(const char *filename)
{
  FILE *f = fopen(filename, "r");
  if (f)
  {
    fclose(f);
    return 1;
  }
  return 0;
}

/*
  Extracts the label from a given filename by removing the directory path,
  file extension, and any suffix after an underscore.
  @param filename The path to the file.
  @return The extracted label as a string.
*/
std::string csvUtil::getLabel(const std::string &filename)
{
  // Remove directory path
  auto slash = filename.find_last_of("/\\");
  std::string base = (slash == std::string::npos) ? filename : filename.substr(slash + 1);

  // Remove file extension
  auto dot = base.find_last_of('.');
  std::string stem = (dot == std::string::npos) ? base : base.substr(0, dot);

  // Get label by removing any suffix after an underscore
  auto us = stem.find('_');
  return (us == std::string::npos) ? stem : stem.substr(0, us);
}

/*
Sets the output filename based on the base path and extractor type.
@param basePath The base path for the output file.
@param extractorType The type of extractor used.
@return The constructed output filename as a string.
*/
std::string csvUtil::setOutputFilename(const std::string &basePath, const ExtractorType &extractorType)
{
  std::string outPath = basePath;
  std::string extractorName = ExtractorFactory::extractorTypeToString(extractorType);
  // append feature name and position to the output file path
  if (outPath.size() >= 4 && outPath.substr(outPath.size() - 4) == ".csv")
    outPath = outPath.substr(0, outPath.size() - 4) + "_" + extractorName + ".csv";
  else
    outPath = outPath + "_" + extractorName + ".csv";
  printf("Using feature type %s\n", extractorName.c_str());
  printf("Output will be saved to %s\n", outPath.c_str());
  return outPath;
}
