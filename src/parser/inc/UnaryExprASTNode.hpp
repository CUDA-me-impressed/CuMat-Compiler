#pragma once

#include "ExprASTNode.hpp"

namespace AST {

enum UNA_OPERATORS { NEG, LNOT, BNOT };

class UnaryExprASTNode : public ExprAST {
    UNA_OPERATORS op;
    std::shared_ptr<ExprAST> operand;

    void codeGen(llvm::Module* module, llvm::IRBuilder<>* Builder,
                 llvm::Function* fp) override;
    void dimensionPass() override;
};
}  // namespace AST