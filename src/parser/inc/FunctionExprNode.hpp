#pragma once

#include <llvm/IR/IRBuilder.h>

#include <string>
#include <vector>

#include "ExprASTNode.hpp"

namespace AST {
class FunctionExprNode : public ExprNode {
   public:
    std::string funcName;
    std::shared_ptr<ExprNode> nonAppliedFunction;
    std::vector<std::shared_ptr<ExprNode>> args;
    void semanticPass() override;
    llvm::Value* codeGen(Utils::IRContext* context) override;
};
}  // namespace AST