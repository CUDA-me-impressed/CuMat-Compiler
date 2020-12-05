#pragma once

#include <vector>

#include "ASTNode.hpp"
#include "ExprASTNode.hpp"

namespace AST {
class BlockNode : public Node {
   public:
    std::vector<std::shared_ptr<Node>> assignments;
    std::shared_ptr<ExprNode> returnExpr;

    llvm::Value* codeGen(llvm::Module* TheModule, llvm::IRBuilder<>* Builder, llvm::Function* fp) override;
};
}  // namespace AST