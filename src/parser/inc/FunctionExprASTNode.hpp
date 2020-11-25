#pragma once

#include <string>
#include <vector>

#include "ExprASTNode.hpp"

namespace AST {
class FunctionExprNode : public ExprNode {
   public:
    std::shared_ptr<ExprNode> nonAppliedFunction;
    std::vector<std::shared_ptr<ExprNode>> args;

    void codeGen(llvm::Module* module) override;
};
}  // namespace AST