#pragma once

#include <string>
#include <vector>

#include "ASTNode.hpp"

namespace AST {
class ImportsNode : public Node {
   public:
    std::vector<std::string> importPaths;
};
}  // namespace AST
