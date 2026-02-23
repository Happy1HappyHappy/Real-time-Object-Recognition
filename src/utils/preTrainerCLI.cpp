/*
Claire Liu, Yu-Jing Wei
preTrainerCLI.cpp

Path: src/utils/preTrainerCLI.cpp
Description: Command line interface for the pre-trainer.
*/

#include "extractorFactory.hpp"
#include "preTrainerCLI.hpp"
#include <getopt.h>
#include <algorithm>
#include <cstdio>

int PreTrainerCLI::parseCLI(
    int argc, char *argv[],
    std::string &dirname,
    ExtractorType &extractorType,
    std::string &outputBase,
    std::string *modelPath)
{
    // Parse command line arguments
    auto args = parse(argc, argv);
    if (args.showHelp)
    {
        printUsage(argv[0]);
        return 0;
    }

    // Check required arguments
    if (args.inputDir.empty() || args.extractorStr.empty() ||
        args.outputPath.empty())
    {
        printf("Error: missing required arguments.\n\n");
        printUsage(argv[0]);
        return -1;
    }

    // get the directory path, extractor type and output file path
    dirname = args.inputDir;
    outputBase = args.outputPath;
    if (modelPath)
    {
        *modelPath = args.modelPath;
    }
    extractorType = ExtractorFactory::stringToExtractorType(args.extractorStr.c_str());
    // Check if the extractor type is valid
    if (extractorType == UNKNOWN_EXTRACTOR)
    {
        printf("Error: unknown extractor type.\n\n");
        printUsage(argv[0]);
        return -1;
    }
    return 0; // Success
}

/*
Parses command line arguments for the pre-trainer.
- @param argc The number of command line arguments.
- @param argv An array of character pointers representing the command line arguments.
- @return An Args struct containing the parsed arguments.
*/
PreTrainerCLI::Args PreTrainerCLI::parse(int argc, char *argv[])
{
    Args args;

    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"extractor", required_argument, 0, 'e'},
        {"output", required_argument, 0, 'o'},
        {"model", required_argument, 0, 'm'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}};

    optind = 1; // reset getopt state

    int opt;
    while ((opt = getopt_long(argc, argv, "i:e:o:m:h", long_options, nullptr)) != -1)
    {
        switch (opt)
        {
        case 'i':
            args.inputDir = optarg;
            break;
        case 'e':
        {
            args.extractorStr = optarg;
            break;
        }
        case 'o':
            args.outputPath = optarg;
            break;
        case 'm':
            args.modelPath = optarg;
            break;
        case 'h':
            args.showHelp = true;
            break;
        default:
            args.showHelp = true;
            break;
        }
    }
    return args;
}

/*
Prints the usage information for the pre-trainer.
- @param prog The name of the program.
*/
void PreTrainerCLI::printUsage(const char *prog)
{
    printf("usage:\n");
    printf("  %s --input <dir> --extractor <type> --output <csv> [--model <onnx>]\n", prog);
    printf("  %s -i <dir> -e <type> -o <csv> [-m <onnx>]\n", prog);
    printf("\n");
    printf("options:\n");
    printf("  -i, --input      <dir>       input image directory\n");
    printf("  -e, --extractor  <type>    baseline | cnn\n");
    printf("  -o, --output     <csv>       output csv path\n");
    printf("  -m, --model      <onnx>      CNN model path (sets RTOR_CNN_MODEL)\n");
    printf("  -h, --help                 show help\n");
}
