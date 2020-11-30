#pragma once

#include <memory>

#include "ExprASTNode.hpp"

namespace Analysis {
class NameTable;
}

namespace AST {
class TernaryExprNode : public ExprNode {
   public:
    std::shared_ptr<ExprNode> condition, truthy, falsey;

    llvm::Value* codeGen(Utils::IRContext* context) override;
    void dimensionPass(Analysis::NameTable* nt) override;
};
}  // namespace AST