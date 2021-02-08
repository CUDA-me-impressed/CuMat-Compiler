#pragma once

#include "ExprASTNode.hpp"

namespace AST {

enum UNA_OPERATORS { NEG, LNOT, BNOT };
static const char* UNA_OP_ENUM_STRING[] = {"neg", "lnot", "bnot"};

class UnaryExprNode : public ExprNode {
   public:
    UNA_OPERATORS op;
    std::shared_ptr<ExprNode> operand;

    llvm::Value* codeGen(Utils::IRContext* context) override;
};
}  // namespace AST