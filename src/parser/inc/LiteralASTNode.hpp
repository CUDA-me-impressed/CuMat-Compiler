#pragma once

#include "ExprASTNode.hpp"

namespace AST {
template <class T>
class LiteralASTNode : public ExprAST {
   public:
    T value;

    void codeGen() override;
};
}  // namespace AST
