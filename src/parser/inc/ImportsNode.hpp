#pragma once

#include "ASTNode.hpp"
#include <vector>
#include <string>

namespace AST {
class ImportsNode : public Node {
   public:
    std::vector<std::string> importPaths;
};
}  // namespace AST
