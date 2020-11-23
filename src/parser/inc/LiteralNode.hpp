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

    llvm::Value* codeGen(Utils::IRContext* context) override;
    void dimensionPass(Analysis::NameTable* nt) override;
};
}  // namespace AST
