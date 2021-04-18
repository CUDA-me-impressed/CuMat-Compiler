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
    void semanticPass(Utils::IRContext* context) override;
    void dimensionPass(Analysis::DimensionSymbolTable* nt) override;
    llvm::Value* codeGen(Utils::IRContext* context) override;
    void printIfMainFunction(Utils::IRContext* context, llvm::Value* returnExprVal);

    [[nodiscard]] std::string toTree(const std::string& prefix, const std::string& childPrefix) const override;
};
}  // namespace AST