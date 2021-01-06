#pragma once

#include <vector>

#include "ASTNode.hpp"
#include "AssignmentNode.hpp"
#include "ExprASTNode.hpp"

namespace AST {
class BlockNode : public Node {
   public:
    std::string callingFunctionName;
    std::vector<std::shared_ptr<AssignmentNode>> assignments;
    std::shared_ptr<ExprNode> returnExpr;

    llvm::Value* codeGen(Utils::IRContext* context) override;
};
}  // namespace AST