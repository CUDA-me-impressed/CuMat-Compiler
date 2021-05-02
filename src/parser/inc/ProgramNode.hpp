#pragma once

#include "ASTNode.hpp"

namespace AST {
class ProgramNode : public Node {
   public:
    llvm::Value* codeGen(Utils::IRContext* context) override;
    [[nodiscard]] std::string toTree(const std::string& prefix, const std::string& childPrefix) const override {
        return std::string{};
    };
    void dimensionPass(Analysis::DimensionSymbolTable* nt) override;
};
}  // namespace AST