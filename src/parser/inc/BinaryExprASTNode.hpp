#pragma once

#include <memory>

#include "ExprASTNode.hpp"

namespace AST {
    enum BIN_OPERATORS {
        PLUS, MINUS, MUL, DIV, LOR, LAND, LT, GT, LTE, GTE, EQ, NEQ, BAND, BOR, POW, MATM
    };

    class BinaryExprASTNode : public ExprAST {
    public:
        std::unique_ptr<ExprAST> lhs, rhs;
        AST::BIN_OPERATORS op;

        void codegen();
    };
}