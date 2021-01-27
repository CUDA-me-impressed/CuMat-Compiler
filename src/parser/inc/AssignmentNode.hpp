#pragma once

#include "ExprASTNode.hpp"
#include "VariableNode.hpp"

namespace AST {
class AssignmentNode : public Node {
   public:
    std::shared_ptr<VariableNode> lVal;
    std::shared_ptr<ExprNode> rVal;

    std::string name;

    llvm::Value* codeGen(Utils::IRContext* context) override;
    [[nodiscard]] std::string toTree(const std::string& prefix, const std::string& childPrefix) const override;
};
}  // namespace AST