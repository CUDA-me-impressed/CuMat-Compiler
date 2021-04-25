#pragma once

#include "ExprASTNode.hpp"

namespace Analysis {
class DimensionSymbolTable;
}

namespace AST {

enum UNA_OPERATORS { NEG, LNOT, BNOT, TRANSPOSE };
static const char* UNA_OP_ENUM_STRING[] = {"neg", "lnot", "bnot", "transpose"};

class UnaryExprNode : public ExprNode {
   public:
    UNA_OPERATORS op;
    std::shared_ptr<ExprNode> operand;
    void semanticPass(Utils::IRContext* context) override;
    llvm::Value* codeGen(Utils::IRContext* context) override;

    bool shouldExecuteGPU(Utils::IRContext* context, UNA_OPERATORS op);
    [[nodiscard]] std::string toTree(const std::string& prefix, const std::string& childPrefix) const override {
        return std::string{};
    };
    void dimensionPass(Analysis::DimensionSymbolTable* nt) override;
};
}  // namespace AST