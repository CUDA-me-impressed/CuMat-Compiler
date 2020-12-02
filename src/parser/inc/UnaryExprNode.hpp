#pragma once

#include "ExprASTNode.hpp"

namespace AST {

enum UNA_OPERATORS { NEG, LNOT, BNOT };

class UnaryExprNode : public ExprNode {
   public:
    UNA_OPERATORS op;
    std::shared_ptr<ExprNode> operand;

    llvm::Value* codeGen(llvm::Module* module, llvm::IRBuilder<>* Builder,
                         llvm::Function* fp) override;
    void recursiveUnaryGeneration(const UNA_OPERATORS& op, llvm::Module* module,
                                  llvm::IRBuilder<>* Builder, llvm::Type* ty,
                                  llvm::AllocaInst* matAlloc,
                                  llvm::Value* opVal,
                                  std::vector<int> dimension, int index = 1,
                                  int prevDim = 1);
};
}  // namespace AST