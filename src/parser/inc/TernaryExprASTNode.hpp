#pragma once

#include <memory>

#include "ExprASTNode.hpp"

namespace AST {
class TernaryExprNode : public ExprNode {
   public:
    std::shared_ptr<ExprNode> condition, truthy, falsey;

    void codeGen(llvm::Module* module) override;
};
}  // namespace AST