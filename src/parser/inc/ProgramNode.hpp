#pragma once

#include "ASTNode.hpp"

namespace AST {
    class ProgramNode : public Node {
    public:
        llvm::Value *codeGen(Utils::IRContext *context) override;
    };
}  // namespace AST