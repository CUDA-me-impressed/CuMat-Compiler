#pragma once

#include "ExprASTNode.hpp"
#include "VariableNode.hpp"

namespace AST {
    class AssignmentNode : public Node {
    public:
        std::shared_ptr<VariableNode> lVal;
        std::shared_ptr<ExprNode> rVal;

        std::string name;

        llvm::Value *codeGen(Utils::IRContext *context) override;
    };
}  // namespace AST