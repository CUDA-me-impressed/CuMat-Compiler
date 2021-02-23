#pragma once

#include <memory>

#include "ExprASTNode.hpp"

namespace AST {
class TernaryExprNode : public ExprNode {
   public:
    std::shared_ptr<ExprNode> condition, truthy, falsey;
    void semanticPass(Utils::IRContext* context) override;
    llvm::Value* codeGen(Utils::IRContext* context) override;
};
}  // namespace AST