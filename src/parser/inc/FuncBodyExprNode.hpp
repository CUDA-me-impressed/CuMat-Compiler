#pragma once

#include <vector>

#include "ExprASTNode.hpp"
#include "FunctionExprNode.hpp"

namespace AST {
class FuncBodyExprNode : public ExprNode {
   public:
    std::shared_ptr<Node> funcSig;  // Function signature

    std::vector<std::shared_ptr<Node>> assignments;
    std::shared_ptr<ExprNode> returnExpr;

    llvm::Value* codeGen(llvm::Module* TheModule, llvm::IRBuilder<>* Builder,
                         llvm::Function* fp) override;
};
}  // namespace AST