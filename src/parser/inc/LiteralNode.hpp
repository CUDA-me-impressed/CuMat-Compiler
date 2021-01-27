#pragma once

#include "ExprASTNode.hpp"

namespace Analysis {
class NameTable;
}

namespace AST {
template <class T>
class LiteralNode : public ExprNode {
   public:
    T value;

    llvm::Value* codeGen(Utils::IRContext* context) override;
    void dimensionPass(Analysis::NameTable* nt) override;
    [[nodiscard]] std::string toTree(const std::string& prefix, const std::string& childPrefix) const override;
};
}  // namespace AST
