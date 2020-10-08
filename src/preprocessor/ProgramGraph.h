#include <memory>
#include <vector>
#include <map>

#ifndef PREPROCESSOR_PROGRAMGRAPH_H
#define PREPROCESSOR_PROGRAMGRAPH_H

namespace Preprocessor {
    class ProgramFileNode {
    public:
        ProgramFileNode(std::string name, std::unique_ptr<std::vector<std::string>> fileContents ) : name(name), fileContents(std::move(fileContents)) {}
    private:
        // Info stored at each node
        std::string name;
        std::unique_ptr<std::vector<std::string>> fileContents;
    };

    class ProgramGraph {
    public:
        ProgramGraph(std::unique_ptr<ProgramFileNode> root) : root(std::move(root)) {}

        // Operations on the graph
        void addInclude(std::unique_ptr<ProgramFileNode> src, std::unique_ptr<ProgramFileNode> dest);
        std::vector<std::unique_ptr<ProgramFileNode>> topologicalSort();
    private:
        std::map<std::unique_ptr<ProgramFileNode>, std::vector<std::unique_ptr<ProgramFileNode>>> vertexEdges;

        // Variables for the graph
        std::unique_ptr<ProgramFileNode> root;
    };
}

#endif
