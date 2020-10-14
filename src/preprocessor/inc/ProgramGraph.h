#include <memory>
#include <vector>
#include <map>
#include <string>

#ifndef PREPROCESSOR_PROGRAMGRAPH_H
#define PREPROCESSOR_PROGRAMGRAPH_H

#define DEFAULT_EXTENSION "cm"

namespace Preprocessor {
    class ProgramGraph;

    class ProgramFileNode {
    public:
        ProgramFileNode(std::string name, std::vector<std::string> & fileContents ) : name(name), fileContents(fileContents) {}
        void expand(std::unique_ptr<ProgramGraph> graph);
        bool canExpand();

        std::vector<std::string> fileContents;

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
        void addInclude(std::shared_ptr<ProgramFileNode> src, std::shared_ptr<ProgramFileNode> dest);
        std::vector<std::shared_ptr<ProgramFileNode>> topologicalSort();
        void generateCompileUnits(std::vector<std::vector<std::shared_ptr<ProgramFileNode>>> & compileUnits);
    private:
        std::vector<std::shared_ptr<ProgramFileNode>> unexpandedNodes;
        std::map<std::shared_ptr<ProgramFileNode>, std::vector<std::shared_ptr<ProgramFileNode>>> vertexEdges;
        std::map<std::string, std::shared_ptr<ProgramFileNode>> fileTable;
        // Variables for the graph
        std::shared_ptr<ProgramFileNode> root;
    };
}

#endif
