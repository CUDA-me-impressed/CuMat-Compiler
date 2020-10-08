#include <memory>
#include <vector>
#include <map>

#ifndef PREPROCESSOR_PROGRAMGRAPH_H
#define PREPROCESSOR_PROGRAMGRAPH_H

namespace Preprocessor {
    class ProgramGraph;

    class ProgramFileNode {
    public:
        ProgramFileNode(std::string name, std::unique_ptr<std::vector<std::string>> fileContents ) : name(name), fileContents(std::move(fileContents)) {}
        void expand(std::unique_ptr<ProgramGraph> graph);
        bool canExpand();

        std::unique_ptr<std::vector<std::string>> fileContents;

    private:
        // Info stored at each node
        bool hasExpanded = false;
        std::string name;
    };

    class ProgramGraph {
    public:
        ProgramGraph(std::shared_ptr<ProgramFileNode> root);

        // Operations on the graph
        void expandAllUnexpanded();
        void addInclude(std::unique_ptr<ProgramFileNode> src, std::unique_ptr<ProgramFileNode> dest);
        std::vector<std::unique_ptr<ProgramFileNode>> topologicalSort();
    private:
        std::vector<std::shared_ptr<ProgramFileNode>> unexpandedNodes;
        std::map<std::unique_ptr<ProgramFileNode>, std::vector<std::unique_ptr<ProgramFileNode>>> vertexEdges;

        // Variables for the graph
        std::shared_ptr<ProgramFileNode> root;
    };
}

#endif
