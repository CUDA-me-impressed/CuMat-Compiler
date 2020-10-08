#include "ProgramGraph.h"
#include <deque>
#include <algorithm>
#include <regex>

/**
 * Constructor for the Program Graph class
 * @param root
 */
Preprocessor::ProgramGraph::ProgramGraph(std::shared_ptr<ProgramFileNode> root) : root(root){
    // Add root node into unexpanded list
    this->unexpandedNodes.push_back(root);
}

/**
 * Adds an include directional edge between the src and dest nodes allowing for us to determine a topological ordering
 * upon the nodes
 * @param src
 * @param dest
 */
void Preprocessor::ProgramGraph::addInclude(std::unique_ptr<ProgramFileNode> src,
                                            std::unique_ptr<ProgramFileNode> dest) {
    // Ensure we have something to push back
    if(!this->vertexEdges.contains(src)){
        this->vertexEdges.insert(std::pair<std::unique_ptr<ProgramFileNode>,
                std::vector<std::unique_ptr<ProgramFileNode>>>(
                        std::move(src),
                        std::vector<std::unique_ptr<ProgramFileNode>>()));
    }
    this->vertexEdges.at(src).push_back(std::move(dest));
}

/**
 * Sorts the graph into a topological ordering based off of the root file node
 * Nodes are merged when they have a single connection to at most one left node
 * @returns An ordered list of the program topology
 */
std::vector<std::unique_ptr<Preprocessor::ProgramFileNode>> Preprocessor::ProgramGraph::topologicalSort() {
    // Implementation of Kahns algorithm
    std::deque<std::unique_ptr<ProgramFileNode>> s;
    std::vector<std::unique_ptr<ProgramFileNode>> l;
    while(!s.empty()){
        auto n = std::move(s.back());
        s.pop_back();
        // Push back new value
        l.push_back(std::move(n));
//        std::vector<std::unique_ptr<ProgramFileNode>> nodes = vertexEdges.at(n);
//        for(int i = 0; i < nodes.size(); i++){
//            auto x = std::move(nodes.at(i));
//            // TODO
//        }

    }

    return std::vector<std::unique_ptr<ProgramFileNode>>();
}

/**
 * Generates an unsorted graph of each of the nodes within the graph
 */
void Preprocessor::ProgramGraph::expandAllUnexpanded() {
    // When we have an unexpanded node (a node which we have not examined the source code for includes)
    // we must expand such that we are able to discover new parts of the program to compile
    while(!unexpandedNodes.empty()){
        // Get the node
        std::shared_ptr<ProgramFileNode> node = unexpandedNodes.at(0);
        // Search for each include
        std::regex rgx(R"(include [_a-zA-Z][_a-zA-Z0-9\-\.]*(\/[_a-zA-Z][_a-zA-Z0-9\-\.]*)*)");
        for(auto it = node->fileContents->begin(); it != node->fileContents->end(); it++){
            // Get the line as a string
            std::smatch matches;
            std::string line = node->fileContents->at(it - node->fileContents->begin());
            if(!std::regex_search(line, matches, rgx)) break;



        }


        this->unexpandedNodes.erase(unexpandedNodes.begin());
    }
}

void Preprocessor::ProgramFileNode::expand(std::unique_ptr<ProgramGraph> graph) {

}
