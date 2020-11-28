#pragma once

#include "ExprASTNode.hpp"

namespace AST {

enum UNA_OPERATORS { NEG, LNOT, BNOT };

class UnaryExprNode : public ExprNode {
    UNA_OPERATORS op;
    std::shared_ptr<ExprNode> operand;

    llvm::Value* codeGen(llvm::Module* module, llvm::IRBuilder<>* Builder,
                         llvm::Function* fp) override;
};
}  // namespace AST