#pragma once

#include "ExprASTNode.hpp"

namespace Analysis {
class NameTable;
}

namespace AST {

enum UNA_OPERATORS { NEG, LNOT, BNOT };

class UnaryExprASTNode : public ExprAST {
    UNA_OPERATORS op;
    std::shared_ptr<ExprAST> operand;

    void codeGen(llvm::Module* module) override;
    void dimensionPass(Analysis::NameTable* nt) override;
};
}  // namespace AST