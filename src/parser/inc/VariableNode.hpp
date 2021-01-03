#pragma once

#include "ExprASTNode.hpp"

namespace AST {
    class VariableNode : public ExprNode {
       public:
        std::shared_ptr<ExprNode> rval;
        std::string name;

        llvm::Value* codegen(Utils::IRContext* context);
    };
}