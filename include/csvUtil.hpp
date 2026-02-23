/*
  Bruce A. Maxwell

  Utility functions for reading and writing CSV files with a specific format

  Each line of the csv file is a filename in the first column, followed by numeric data for the remaining columns
  Each line of the csv file has to have the same number of columns
 */

#ifndef CVS_UTIL_H
#define CVS_UTIL_H

#include <vector>
#include "extractorFactory.hpp"
class csvUtil
{
public:
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
  static int append_image_data_csv(const char *filename, const char *image_filename, std::vector<float> &image_data, int reset_file = 0);

  /*
  Clears the contents of the specified file.
  @param filename The path to the file to be cleared.
  @return 0 on success, non-zero value on error.
  */
  static int clearExistingFile(const char *filename);

  /*
  Checks if the specified file exists.
  @param filename The path to the file to be checked.
  @return 1 if the file exists, 0 otherwise.
  */
  static int fileExists(const char *filename);

  /*
  Extracts the label from a given filename by removing the directory path,
  file extension, and any suffix after an underscore.
  @param filename The path to the file.
  @return The extracted label as a string.
  */
  static std::string getLabel(const std::string &filename);

  /*
  Sets the output filename based on the base path and extractor type.
  @param basePath The base path for the output file.
  @param extractorType The type of extractor used.
  @return The constructed output filename as a string.
  */
  static std::string setOutputFilename(const std::string &basePath, const ExtractorType &extractorType);
};
#endif
