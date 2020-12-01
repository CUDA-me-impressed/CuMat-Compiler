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

    llvm::Value* codeGen(llvm::Module* module, llvm::IRBuilder<>* Builder, llvm::Function* fp) override;
};
}  // namespace AST