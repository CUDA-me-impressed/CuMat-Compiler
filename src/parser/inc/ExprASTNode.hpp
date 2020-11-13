#pragma once

#include "ASTNode.hpp"

class ExprAST : public ASTNode {
public:
    virtual ~ExprAST() {}
    virtual void codegen();
};