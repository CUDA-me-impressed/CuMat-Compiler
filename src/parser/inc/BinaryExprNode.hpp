#pragma once

#include <llvm/IR/IRBuilder.h>

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

class BinaryExprNode : public ExprNode {
   public:
    std::shared_ptr<ExprNode> lhs, rhs;
    AST::BIN_OPERATORS op;

    llvm::Value* codeGen(llvm::Module* TheModule, llvm::IRBuilder<>* Builder,
                         llvm::Function* fp) override;
};
}  // namespace AST