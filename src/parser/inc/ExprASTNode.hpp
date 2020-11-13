#pragma once

#include "ASTNode.hpp"

namespace AST {
class ExprAST : public Node {
   public:
    virtual ~ExprAST() {}
    virtual void codegen();
};
}  // namespace AST
