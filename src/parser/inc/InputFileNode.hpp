#pragma once

#include <llvm/IR/IRBuilder.h>

#include <string>
#include <vector>

#include "ExprASTNode.hpp"

namespace Analysis {
class DimensionSymbolTable;
}

namespace AST {
class InputFileNode : public ExprNode {
   public:
    std::string fileName{};

    void semanticPass(Utils::IRContext* context) override;
    void dimensionPass(Analysis::DimensionSymbolTable* nt) override;
    llvm::Value* codeGen(Utils::IRContext* context) override;
    [[nodiscard]] std::string toTree(const std::string& prefix, const std::string& childPrefix) const override;
};
}  // namespace AST