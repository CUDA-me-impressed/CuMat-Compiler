#pragma once

#include "ExprASTNode.hpp"

namespace Analysis {
class NameTable;
}

namespace AST {

enum UNA_OPERATORS { NEG, LNOT, BNOT };
static const char* UNA_OP_ENUM_STRING[] = {"neg", "lnot", "bnot"};

class UnaryExprNode : public ExprNode {
   public:
    UNA_OPERATORS op;
    std::shared_ptr<ExprNode> operand;

    llvm::Value* codeGen(Utils::IRContext* context) override;
    void dimensionPass(Analysis::NameTable* nt) override;
    [[nodiscard]] std::string toTree(const std::string& prefix, const std::string& childPrefix) const override;
};
}  // namespace AST