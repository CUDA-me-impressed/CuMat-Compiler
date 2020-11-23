#pragma once

#include <llvm-10/llvm/IR/IRBuilder.h>

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

    llvm::Value* codeGen(llvm::Module* TheModule, llvm::IRBuilder<>* Builder,
                         llvm::Function* fp) override;
    void dimensionPass() override;
};
}  // namespace AST