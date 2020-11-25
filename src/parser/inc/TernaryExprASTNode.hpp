#pragma once

#include <memory>

#include "ExprASTNode.hpp"

namespace AST {
class TernaryExprASTNode : public ExprAST {
   public:
    std::shared_ptr<ExprAST> condition, truthy, falsey;

    llvm::Value* codeGen(llvm::Module* module, llvm::IRBuilder<>* Builder,
                         llvm::Function* fp) override;
};
}  // namespace AST