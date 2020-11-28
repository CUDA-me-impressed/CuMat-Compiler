#pragma once

#include "ExprASTNode.hpp"

namespace AST {
template <class T>
class LiteralNode : public ExprNode {
   public:
    T value;

    void codeGen(llvm::Module* module) override;
};
}  // namespace AST
