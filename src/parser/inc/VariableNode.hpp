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
    void semanticPass() override;
    llvm::Value* codeGen(Utils::IRContext* context) override;
};
}  // namespace AST