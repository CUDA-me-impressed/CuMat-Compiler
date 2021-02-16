#pragma once

#include <string>
#include <vector>

#include "ASTNode.hpp"

namespace AST {
class DecompNode : public Node {
   public:
    std::string lVal;
    std::variant<std::string, std::shared_ptr<DecompNode>> rVal;
    llvm::Value* codeGen(Utils::IRContext* context) override;

    void semanticPass() override;
};
}  // namespace AST
