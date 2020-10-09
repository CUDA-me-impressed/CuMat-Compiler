#pragma once

#include <string>

enum class WARNINGS { ALL, VERBOSE, INFO, NONE };

enum class OPTIMISATION { NONE, ALL, EXPERIMENTAL };

class CompilerOptions {
   public:
    WARNINGS warningVerbosity = WARNINGS::NONE;

    OPTIMISATION optimisationLevel = OPTIMISATION::ALL;

    std::string inputFile;

    std::string outputFile;  // Default to name of file

    CompilerOptions(const std::string& inpFile)
        : inputFile(inpFile), outputFile(inpFile) {}
};