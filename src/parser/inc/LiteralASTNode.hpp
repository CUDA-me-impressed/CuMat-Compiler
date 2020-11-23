#pragma once

#include "ExprASTNode.hpp"

namespace AST {
template <class T>
class LiteralASTNode : public ExprAST {
   public:
    T value;

    llvm::Value* codeGen(llvm::Module* module, llvm::IRBuilder<>* Builder,
                         llvm::Function* fp) override;
};
}  // namespace AST
