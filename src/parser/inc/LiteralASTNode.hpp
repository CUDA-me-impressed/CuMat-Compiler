#pragma once

#include "ExprASTNode.hpp"

namespace Analysis {
class NameTable;
}

namespace AST {
template <class T>
class LiteralASTNode : public ExprAST {
   public:
    T value;

    void codeGen(llvm::Module* module) override;
    void dimensionPass(Analysis::NameTable* nt) override;
};
}  // namespace AST
