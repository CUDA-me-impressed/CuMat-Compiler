#pragma once

#include <string>
#include <utility>

enum class WARNINGS { ALL, INFO, NONE };

enum class OPTIMISATION { NONE, ALL, EXPERIMENTAL };

enum class COMPUTATION {AUTO, GPU, CPU};

class CompilerOptions {
   public:
    WARNINGS warningVerbosity = WARNINGS::NONE;

    OPTIMISATION optimisationLevel = OPTIMISATION::ALL;

    COMPUTATION computationMode = COMPUTATION::AUTO;

    std::string inputFile;

    std::string outputFile;  // Default to name of file

    bool silent = false;

    CompilerOptions() = default;

    explicit CompilerOptions(const std::string& inpFile) : inputFile(inpFile), outputFile(inpFile) {}

    CompilerOptions(std::string inpFile, std::string outFile)
        : inputFile(std::move(inpFile)), outputFile(std::move(outFile)) {}
};