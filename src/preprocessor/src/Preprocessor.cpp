#include "Preprocessor.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>

#include "ProgramGraph.hpp"

/**
 * Function to load all of the various files we can discover
 * This is done either from a known dictionary of include / path pairs or
 * assumed as from pwd.
 * @return
 */
std::vector<std::string> Preprocessor::SourceFileLoader::load() {
    if (lookupPath.empty()) {
        // We will load the files from the current directory
        auto fileLines = load(this->rootFile);

        // Generate program graph and construct compile unit
        // What is a compile unit? A compile unit is a group of code that we can
        // consider in isolation from the other files. This allows for us to
        // compile each unit in parallel which will assist in the overall
        // compile speed
        std::shared_ptr<ProgramFileNode> rootNode = std::make_shared<ProgramFileNode>(rootFile, *fileLines.get());
        std::unique_ptr<ProgramGraph> program = std::make_unique<ProgramGraph>(rootNode);
        // We should have the entire program loaded into memory now, let us sort
        // the graph
        std::vector<std::vector<std::shared_ptr<ProgramFileNode>>> compileUnits;
        program->generateCompileUnits(compileUnits);
    } else {
        // Assume we have a lookup of the various file paths
    }

    return std::vector<std::string>();
}

std::unique_ptr<std::vector<std::string>> Preprocessor::SourceFileLoader::load(const std::string& file) {
    // We will load the files from the current directory
    std::ifstream fileStream(file);
    std::unique_ptr<std::vector<std::string>> fileLines = std::make_unique<std::vector<std::string>>();
    if (!fileStream.is_open()) {
        std::ostringstream ss;
        ss << "Could not find source file defined at [" << file << " ] in "
           << std::experimental::filesystem::current_path();
        throw std::experimental::filesystem::filesystem_error(ss.str(), std::error_code(15, std::system_category()));
    }

    // Load in the root file into a vector
    std::string line;
    while (std::getline(fileStream, line)) fileLines->push_back(line);

    // Close The File
    fileStream.close();
    return fileLines;
}
