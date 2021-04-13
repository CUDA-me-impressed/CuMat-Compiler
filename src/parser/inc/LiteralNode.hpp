#pragma once

#include "ExprASTNode.hpp"

namespace Analysis {
class DimensionSymbolTable;
}

namespace AST {
template <class T>
class LiteralNode : public ExprNode {
   public:
    T value;

    void semanticPass(Utils::IRContext* context) override;
    llvm::Value* codeGen(Utils::IRContext* context) override;
    void dimensionPass(Analysis::DimensionSymbolTable* nt) override;
    [[nodiscard]] std::string toTree(const std::string& prefix, const std::string& childPrefix) const override;
};
}  // namespace AST
