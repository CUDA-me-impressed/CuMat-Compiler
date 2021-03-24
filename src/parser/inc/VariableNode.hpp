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
};
}  // namespace AST