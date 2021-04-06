#pragma once

#include "DecompNode.hpp"
#include "ExprASTNode.hpp"
#include "VariableNode.hpp"

namespace AST {
class AssignmentNode : public Node {
   public:
    // Note, EITHER name or lVal will be filled in. This could be changed if it would be easier. But I'm not sure what
    // would be.
    std::string name;
    std::shared_ptr<DecompNode> lVal;
    std::shared_ptr<ExprNode> rVal;

    void semanticPass(Utils::IRContext* context) override;
    llvm::Value* codeGen(Utils::IRContext* context) override;
    llvm::Value* decompAssign(Utils::IRContext* context, std::shared_ptr<DecompNode> decompNode,
                              llvm::Value* matHeader);
};
}  // namespace AST