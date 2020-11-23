#pragma once

#include <llvm-10/llvm/IR/IRBuilder.h>

#include <string>
#include <vector>

#include "ExprASTNode.hpp"

namespace AST {
class FunctionExprASTNode : public ExprAST {
   public:
    const std::string funcName;
    std::vector<std::shared_ptr<ExprAST>> args;

    void codeGen(llvm::Module* module, llvm::IRBuilder<>* Builder,
                 llvm::Function* fp) override;
};
}  // namespace AST