#pragma once

#include "ExprASTNode.hpp"

namespace AST {
class VariableNode : public ExprNode {
   public:
    std::string name;
};
}  // namespace AST