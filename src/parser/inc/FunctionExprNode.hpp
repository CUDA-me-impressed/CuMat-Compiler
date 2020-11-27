#pragma once

#include <llvm/IR/IRBuilder.h>

#include <string>
#include <vector>

#include "ExprASTNode.hpp"

namespace AST {
class FunctionExprNode : public ExprAST {
   public:
    std::shared_ptr<ExprAST> nonAppliedFunction;
    std::vector<std::shared_ptr<ExprAST>> args;

    llvm::Value* codeGen(llvm::Module* module, llvm::IRBuilder<>* Builder,
                         llvm::Function* fp) override;
};
}  // namespace AST