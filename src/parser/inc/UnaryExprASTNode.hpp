#pragma once

#include "ExprASTNode.hpp"

namespace AST {

    enum UNA_OPERATORS {
        NEG, LNOT, BNOT
    };

    class UnaryExprASTNode : public ExprAST {
        std::shared_ptr<ExprAST> operand;
    };
}