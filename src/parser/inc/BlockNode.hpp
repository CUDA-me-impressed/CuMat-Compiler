#pragma once

#include <vector>

#include "ASTNode.hpp"
#include "ExprASTNode.hpp"

namespace AST {
class BlockNode : public Node {
   public:
    std::vector<std::shared_ptr<Node>> assignments;
    std::shared_ptr<ExprNode> returnExpr;

    llvm::Value* codeGen(Utils::IRContext * context) override;
};
}  // namespace AST