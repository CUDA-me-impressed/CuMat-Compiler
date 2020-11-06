#include "ProgramGraph.hpp"

#include <algorithm>
#include <deque>
#include <iostream>
#include <regex>
#include <set>

#include "Preprocessor.hpp"

/**
 * Constructor for the Program Graph class
 * @param root
 */
Preprocessor::ProgramGraph::ProgramGraph(std::shared_ptr<ProgramFileNode> root)
    : root(root) {
    // Add root node into unexpanded list
    this->unexpandedNodes.push_back(root);
    expandAllUnexpanded();
}

/**
 * Adds an include directional edge between the src and dest nodes allowing for
 * us to determine a topological ordering upon the nodes
 * @param src
 * @param dest
 */
void Preprocessor::ProgramGraph::addInclude(
    std::shared_ptr<ProgramFileNode> src,
    std::shared_ptr<ProgramFileNode> dest) {
    // operator[] applied to map auto creates the vector if it does not exist
    this->vertexEdges[src].push_back(dest);
    this->vertexEdgesReverse[dest].push_back(src);
}

/**
 * Sorts the graph into a topological ordering based off of the root file node
 * Nodes are merged when they have a single connection to at most one left node
 * @returns An ordered list of the program topology
 */
std::vector<std::shared_ptr<Preprocessor::ProgramFileNode>>
Preprocessor::ProgramGraph::topologicalSort() {
    // Implementation of Kahns algorithm
    std::deque<std::shared_ptr<ProgramFileNode>> s;
    std::vector<std::shared_ptr<ProgramFileNode>> l;
    // Root node will be the only node with no incoming edges by def
    s.push_back(root);
    while (!s.empty()) {
        auto n = s.front();
        s.pop_front();
        l.push_back(n);
        // For all nodes with an edge from n to m
        for (auto m : this->vertexEdges[n]) {
            s.push_back(m);
        }
    }
    // Return a value, its just a vector of pointers, should be light weight
    // enough
    return l;
}

/**
 * Generates an unsorted graph of each of the nodes within the graph
 */
void Preprocessor::ProgramGraph::expandAllUnexpanded() {
    // When we have an unexpanded node (a node which we have not examined the
    // source code for includes) we must expand such that we are able to
    // discover new parts of the program to compile
    while (!unexpandedNodes.empty()) {
        // Get the node
        std::shared_ptr<ProgramFileNode> node = unexpandedNodes.at(0);
        // Search for each include, this does not deal with comments TODO
        std::regex rgx(
            R"(import ([_a-zA-Z][_a-zA-Z0-9\-\.]*(\/[_a-zA-Z][_a-zA-Z0-9\-\.]*)*))");
        for (auto it = node->fileContents.begin();
             it != node->fileContents.end(); it++) {
            // Get the line as a string
            std::smatch matches;
            std::string line =
                node->fileContents.at(it - node->fileContents.begin());
            if (!std::regex_search(line, matches, rgx)) break;
            // Our regex splits the import into a number of capture groups
            // Group 1 consists of the full path file, minus the extension
            // Nice and simple, let us now find and iteratively add the
            // file to our Program Graph.
            std::string included = matches[1].str() + "." + DEFAULT_EXTENSION;
            // If we have already added, don't bother doing again (guarantees no
            // cycles)
            if (fileTable.count(included)) continue;
            auto newFileLines = Preprocessor::SourceFileLoader::load(included);
            std::shared_ptr<ProgramFileNode> newFileNode =
                std::make_shared<ProgramFileNode>(included,
                                                  *newFileLines.get());
            fileTable[included] = newFileNode;
            // We need to handle this new node by adding it to the program graph
            this->addInclude(node, newFileNode);
        }
        this->unexpandedNodes.erase(unexpandedNodes.begin());
    }
}

void Preprocessor::ProgramGraph::generateCompileUnits(
    std::vector<std::vector<std::shared_ptr<ProgramFileNode>>> &compileUnits) {
    auto sortedProgram = topologicalSort();
    std::reverse(sortedProgram.begin(), sortedProgram.end());
    int i = 0;
    int layer = 0;
    std::set<std::shared_ptr<ProgramFileNode>> addedNodes;
    // Continue until we have each node added into our compile units
    compileUnits.emplace_back();
    while (i != sortedProgram.size()) {
        // Push back any node to store for later
        for (auto n : vertexEdgesReverse[sortedProgram.at(i)]) {
            addedNodes.emplace(n);
        }
        if (std::find(addedNodes.begin(), addedNodes.end(),
                      sortedProgram.at(i)) != addedNodes.end()) {
            compileUnits.emplace_back();
            layer++;
            compileUnits[layer].push_back(sortedProgram.at(i));

        } else {
            compileUnits[layer].push_back(sortedProgram.at(i));
        }
        i++;
    }
}
