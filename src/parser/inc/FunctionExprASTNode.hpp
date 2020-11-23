#pragma once

#include <llvm-10/llvm/IR/IRBuilder.h>

#include <string>
#include <vector>

#include "ExprASTNode.hpp"

namespace AST {
class FunctionExprASTNode : public ExprAST {
   public:
    std::shared_ptr<ExprAST> nonAppliedFunction;
    std::vector<std::shared_ptr<ExprAST>> args;

    llvm::Value* codeGen(llvm::Module* module, llvm::IRBuilder<>* Builder,
                         llvm::Function* fp) override;
    void dimensionPass() override;
};
}  // namespace AST