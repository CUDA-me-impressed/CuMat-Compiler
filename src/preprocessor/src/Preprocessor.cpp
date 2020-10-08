#include "Preprocessor.h"
#include "ProgramGraph.h"

#include <fstream>
#include <iostream>
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
        std::unique_ptr<std::vector<std::string>> fileLines = std::make_unique<std::vector<std::string>>();
        std::string line;
        if(!rootStream.is_open()){
            std::cerr << "Could not open file" << std::endl;
        }

        // Load in the root file into a vector
        while (std::getline(rootStream, line)) fileLines->push_back(line);
        //Close The File
        rootStream.close();
        // Generate program graph and construct compile unit
        // What is a compile unit? A compile unit is a group of code that we can consider in isolation from the other
        // files. This allows for us to compile each unit in parallel which will assist in the overall compile speed
        std::shared_ptr<ProgramFileNode> rootNode = std::make_shared<ProgramFileNode>(rootFile, std::move(fileLines));
        std::unique_ptr<ProgramGraph> program = std::make_unique<ProgramGraph>(rootNode);

    }else{
        // Assume we have a lookup of the various file paths
    }

    return std::vector<std::string>();
}

