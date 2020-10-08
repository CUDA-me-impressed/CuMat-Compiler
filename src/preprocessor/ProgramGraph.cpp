#include "ProgramGraph.h"
#include <deque>
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
