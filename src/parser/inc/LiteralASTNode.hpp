#pragma once

#include "ExprASTNode.hpp"

namespace AST {
template <class T>
class LiteralASTNode : public ExprAST {
   public:
    T value;

    void codeGen(llvm::Module* module, llvm::Function * fp) override;
};
}  // namespace AST
