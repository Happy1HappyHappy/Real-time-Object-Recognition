/*
  Claire Liu, Yu-Jing Wei
  preTrainerCLI.hpp

  Path: include/preTrainerCLI.hpp
  Description: Header file for preTrainerCLI.cpp to parse command-line
                arguments for pre-training.
*/

#include <getopt.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>

/*
PreTrainerCLI class to parse command-line arguments for pre-training.
Struct Args:
    - inputDir: The directory containing input images.
    - extractorStr: The extractor type to use.
    - outputPath: The path to save the extracted features.
    - showHelp: A flag indicating whether to display the help message.
public:
    - parse(int argc, char *argv[]): Parses the command-line arguments and returns an Args struct.
    - printUsage(const char *prog): Prints the usage information for the program.
*/
class PreTrainerCLI
{
public:
    struct Args
    {
        std::string inputDir;
        std::string extractorStr;
        std::string outputPath;
        bool showHelp = false;
    };

    static int parseCLI(
        int argc, char *argv[],
        std::string &dirname,
        ExtractorType &extractorType,
        std::string &outputBase);
    static Args parse(int argc, char *argv[]);
    static void printUsage(const char *prog);
};
