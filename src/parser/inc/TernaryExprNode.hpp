#pragma once

#include <memory>

#include "ExprASTNode.hpp"

namespace AST {
class TernaryExprNode : public ExprNode {
   public:
    std::shared_ptr<ExprNode> condition, truthy, falsey;

    llvm::Value* codeGen(llvm::Module* module, llvm::IRBuilder<>* Builder, llvm::Function* fp) override;
};
}  // namespace AST