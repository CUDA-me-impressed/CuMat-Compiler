#pragma once

#include <vector>

#include "ASTNode.hpp"

namespace AST {
class SliceNode : public Node {
   public:
    std::vector<std::variant<bool, std::vector<int>>> slices;

    llvm::Value* codeGen(Utils::IRContext* context) override;
};
}  // namespace AST