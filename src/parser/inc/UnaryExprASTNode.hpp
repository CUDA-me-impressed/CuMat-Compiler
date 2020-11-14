#pragma once

#include "ExprASTNode.hpp"

namespace AST {

    enum UNA_OPERATORS {
        NEG, LNOT, BNOT
    };

    class UnaryExprASTNode : public ExprAST {
        UNA_OPERATORS op;
        std::unique_ptr<ExprAST> operand;
    };
}