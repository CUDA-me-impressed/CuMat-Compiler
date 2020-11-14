#pragma once

#include "ExprASTNode.hpp"

namespace AST {

    enum UNA_OPERATORS {

    };

    class UnaryExprASTNode : public ExprAST {
        std::unique_ptr<ExprAST> operand;

    };
}