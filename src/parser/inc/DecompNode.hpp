#pragma once

#include "ASTNode.hpp"
#include <vector>
#include <string>

namespace AST {
class DecompNode : public Node {
   public:
    std::string lVal;
    std::variant<std::string,std::shared_ptr<DecompNode>> rVal;
};
}  // namespace AST
