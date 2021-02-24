#pragma once

#include <llvm/IR/IRBuilder.h>

#include <string>
#include <vector>

#include "ExprASTNode.hpp"

namespace Analysis {
class NameTable;
}

namespace AST {
class FunctionExprNode : public ExprNode {
   public:
    std::string funcName{};
    std::shared_ptr<ExprNode> nonAppliedFunction{};
    std::vector<std::shared_ptr<ExprNode>> args;

    void dimensionPass(Analysis::NameTable* nt) override{};
    llvm::Value* codeGen(Utils::IRContext* context) override;
    [[nodiscard]] std::string toTree(const std::string& prefix, const std::string& childPrefix) const override;
};
}  // namespace AST