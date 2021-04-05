#pragma once

#include <llvm/IR/IRBuilder.h>

#include <memory>

#include "ExprASTNode.hpp"

namespace AST {
enum BIN_OPERATORS { PLUS, MINUS, MUL, DIV, LOR, LAND, LT, GT, LTE, GTE, EQ, NEQ, BAND, BOR, POW, MATM, CHAIN };
static const char* BIN_OP_ENUM_STRING[] = {"plus", "minus", "mul", "div",  "lor", "land", "lt",   "gt",   "lte",
                                           "gte",  "eq",    "neq", "band", "bor", "pow",  "matm", "chain"};

class BinaryExprNode : public ExprNode {
   public:
    std::shared_ptr<ExprNode> lhs, rhs;
    AST::BIN_OPERATORS op;

    llvm::Value* applyOperatorToOperands(Utils::IRContext* context, const AST::BIN_OPERATORS& op, llvm::Value* lhs,
                                         llvm::Value* rhs, const std::string& name = "");
    llvm::Value* applyPowerToOperands(Utils::IRContext* context, llvm::Value* lhs, llvm::Value* rhs, const bool isFloat,
                                      const std::string& name);
    void semanticPass(Utils::IRContext* context) override;
    llvm::Value* codeGen(Utils::IRContext* context) override;
    // Operation specific codegen
    llvm::Value* elementWiseCodeGen(Utils::IRContext* context, llvm::Value* lhsVal, llvm::Value* rhsVal,
                                    const Typing::MatrixType& lhsType, const Typing::MatrixType& rhsType,
                                    llvm::Instruction* matAlloc, const Typing::MatrixType& resType);

    llvm::Value* matrixMultiply(Utils::IRContext* context, std::shared_ptr<Typing::MatrixType> lhsMat,
                                std::shared_ptr<Typing::MatrixType> rhsMat, llvm::Value* lhsVal, llvm::Value* rhsVal);

    bool shouldExecuteGPU(Utils::IRContext* context, BIN_OPERATORS op) const;
};
}  // namespace AST