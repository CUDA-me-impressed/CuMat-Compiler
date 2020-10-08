#include "Preprocesor.h"
#include "ProgramGraph.h"

#include <fstream>

/**
 * Function to load all of the various files we can discover
 * This is done either from a known dictionary of include / path pairs or assumed as
 * from pwd.
 * @return
 */
std::vector<std::string> Preprocessor::SourceFileLoader::load() {
    if(lookupPath.empty()){
        // We will load the files from the current directory
        std::ifstream rootStream(this->rootFile);
        std::vector<std::string> program;
        std::string line;
        // Load in the root file into a vector
        while (std::getline(rootStream, line)) program.push_back(line);
        // Generate program graph and construct compile unit
        // What is a compile unit? A compile unit is a group of code that we can consider in isolation from the other
        // files. This allows for us to compile each unit in parallel which will assist in the overall compile speed
        // TODO
        return program;
    }else{
        // Assume we have a lookup of the various file paths
    }

    return std::vector<std::string>();
}

