#pragma once

#include "ExprASTNode.hpp"

namespace Analysis {
class NameTable;
}

namespace AST {
template <class T>
class LiteralNode : public ExprNode {
   public:
    T value;

    llvm::Value* codeGen(llvm::Module* module, llvm::IRBuilder<>* Builder,
                         llvm::Function* fp) override;
    void dimensionPass(Analysis::NameTable* nt) override;
};
}  // namespace AST
