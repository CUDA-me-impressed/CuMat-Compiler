#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <algorithm>

#include "CompilerOptions.hpp"

void printArgumentError(std::string message, std::string arg)
{
    const std::string helpText = "CuMat Compiler Syntax: CuMat inputFile [ -Wall | -Winfo | -Wnone ] [ -Oall | -Onone | -Oexp ] [ -o outputfile ]";
    std::cout << helpText << std::endl;
    std::cout << message << arg << std::endl;
}

int main(int argc, char* argv[], char* envp[])
{
    std::vector<std::string> args;
    std::string inputFileName;
    std::string outputFile;

    const std::set<std::string> validFlags = { "-Wall", "-Winfo", "-Wnone", "-Oall", "-Onone", "-Oexp", "-o" };

    //First argument is always name of exe, ignore
    for (int i = 1; i < argc; ++i)
    {
        auto arg = std::string(argv[i]);
        if (validFlags.find(arg) != validFlags.end())
        {
            args.emplace_back(arg);
        }
        else if (!args.empty() && args.back() == "-o" && outputFile.empty()) //Output and at least one additional
        {
            outputFile = arg;
        }
        else if (inputFileName.empty())
        {
            inputFileName = arg;
        }
        else
        {
            printArgumentError("Unrecognised argument", arg);
            return 1; //Quit early
        }

    }

    if (inputFileName.empty())
    {
        printArgumentError("Needs an input file, e.g. ", "main.cm");
        return 1; //Quit early
    }

    CompilerOptions co;
    if (outputFile.empty())
    {
        co = CompilerOptions(inputFileName);
    }
    else
    {
        co = CompilerOptions(inputFileName, outputFile);
    }

    //Assign Warning level
    std::vector<std::string> warningLevels;
    std::copy_if(args.begin(), args.end(), std::back_inserter(warningLevels), [](std::string s)
    { return s.rfind("-W", 0) == 0; });
    if (warningLevels.size() > 1)
    {
        printArgumentError("Maximum of one warning level to be set", "");
        return 1;
    }
    if (warningLevels.size() == 1)
    {
        auto warn = warningLevels.front();
        if (warn == "-Wall")
        {
            co.warningVerbosity = WARNINGS::ALL;
        }
        else if (warn == "-Winfo")
        {
            co.warningVerbosity = WARNINGS::INFO;
        }
        else if (warn == "-Wnone")
        {
            co.warningVerbosity = WARNINGS::NONE;
        }
        else
        {    //This shouldn't happen, but for completeness
            printArgumentError("Unrecognised warning level", warn);
            return 1; //Quit early
        }
    }

    //Assign Optimisation level
    std::vector<std::string> optimiseLevels;
    std::copy_if(args.begin(), args.end(), std::back_inserter(optimiseLevels), [](std::string s)
    { return s.rfind("-O", 0) == 0; });
    if (optimiseLevels.size() > 1)
    {
        printArgumentError("Maximum of one optimisation level to be set", "");
        return 1;
    }
    if (optimiseLevels.size() == 1)
    {
        auto opt = optimiseLevels.front();
        if (opt == "-Oall")
        {
            co.optimisationLevel = OPTIMISATION::ALL;
        }
        else if (opt == "-Onone")
        {
            co.optimisationLevel = OPTIMISATION::NONE;
        }
        else if (opt == "-Oexp")
        {
            co.optimisationLevel = OPTIMISATION::EXPERIMENTAL;
        }
        else
        {    //This shouldn't happen, but for completeness
            printArgumentError("Unrecognised optimisation level", opt);
            return 1; //Quit early
        }
    }

    std::cout << "Input file: " << co.inputFile << std::endl;
    std::cout << "Output file: " << co.outputFile << std::endl;
    for (auto arg : args)
    {
        std::cout << arg << " argument applied" << std::endl;
    }
    return 0;
}
