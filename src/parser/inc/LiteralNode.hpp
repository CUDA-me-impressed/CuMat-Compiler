#pragma once

#include "ExprASTNode.hpp"

namespace AST {
template <class T>
class LiteralNode : public ExprNode {
   public:
    T value;

    void semanticPass() override;
    llvm::Value* codeGen(Utils::IRContext* context) override;
};
}  // namespace AST
