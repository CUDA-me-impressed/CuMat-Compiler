#pragma once

#include <memory>

#include "ExprASTNode.hpp"

namespace AST {
enum BIN_OPERATORS {
    PLUS,
    MINUS,
    MUL,
    DIV,
    LOR,
    LAND,
    LT,
    GT,
    LTE,
    GTE,
    EQ,
    NEQ,
    BAND,
    BOR,
    POW,
    MATM,
    CHAIN
};

class BinaryExprASTNode : public ExprAST {
   public:
    std::shared_ptr<ExprAST> lhs, rhs;
    AST::BIN_OPERATORS op;

    void codegen();
};
}  // namespace AST