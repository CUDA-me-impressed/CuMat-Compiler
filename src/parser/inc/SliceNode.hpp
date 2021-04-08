#pragma once

#include <vector>

#include "ASTNode.hpp"

namespace AST {
class SliceNode : public Node {
   public:
    std::vector<std::variant<bool, std::vector<int>>> slices;

    void semanticPass(Utils::IRContext* context) override;

    llvm::Value* codeGen(Utils::IRContext* context) override;
    [[nodiscard]] std::string toTree(const std::string& prefix, const std::string& childPrefix) const override{};
};
}  // namespace AST