#pragma once

#include "ExprASTNode.hpp"

enum class TYPE {
    STRING, INT, FLOAT
};

template<class T>
class LiteralASTNode : public ExprAST {
public:
    T value;

    void codegen();
};