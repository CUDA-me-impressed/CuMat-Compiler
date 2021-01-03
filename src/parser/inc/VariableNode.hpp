#pragma once

#include "ExprASTNode.hpp"

namespace AST {
    class VariableNode : public ExprNode {
        std::string name;
    };
}