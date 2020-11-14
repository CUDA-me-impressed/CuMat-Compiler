#pragma once

#include <memory>

#include "ExprASTNode.hpp"

namespace AST {
    enum OPERATORS {
        PLUS, MINUS, MUL, DIV, LT, GT, LTE, GTE, EQ, NEQ, BAND, BOR, POW, MATM
    };

    class BinaryExprASTNode : public ExprAST {
    public:
        std::unique_ptr<ExprAST> lhs, rhs;
        AST::OPERATORS op;

        void codegen();
    };
}