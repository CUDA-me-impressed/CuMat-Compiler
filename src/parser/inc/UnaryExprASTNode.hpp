#pragma once

#include "ExprASTNode.hpp"

namespace AST {

enum UNA_OPERATORS { NEG, LNOT, BNOT };

class UnaryExprNode : public ExprNode {
    UNA_OPERATORS op;
    std::shared_ptr<ExprNode> operand;

    void codeGen(llvm::Module* module) override;
};
}  // namespace AST