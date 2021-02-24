#pragma once

#include <vector>

#include "ExprASTNode.hpp"
#include "SliceNode.hpp"

namespace AST {
class VariableNode : public ExprNode {
   public:
    std::vector<std::string> namespacePath;
    std::string name;
    std::shared_ptr<SliceNode> variableSlicing;
    void semanticPass(Utils::IRContext* context) override;
    llvm::Value* codeGen(Utils::IRContext* context) override;

    llvm::Value* handleSlicing(Utils::IRContext* context, llvm::Value* val);
    [[nodiscard]] std::string toTree(const std::string& prefix, const std::string& childPrefix) const override;
};
}  // namespace AST