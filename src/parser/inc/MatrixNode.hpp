#pragma once

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>

#include <iostream>
#include <vector>

#include "ExprASTNode.hpp"
#include "LiteralNode.hpp"
#include "Type.hpp"

namespace Analysis {
class DimensionSymbolTable;
}

namespace AST {
class MatrixNode : public ExprNode {
   private:
    void dim_subpass(std::vector<uint>& apparent_dim, std::vector<uint>& size, const std::vector<uint>& nodedims,
                     int sep);

   public:
    std::vector<std::shared_ptr<ExprNode>> data;
    std::vector<uint> separators;  // the list of seperator kinds between each data element. Invalid after dimension
                                   // pass (due to matrix lifting)

    std::vector<uint> getDimensions();
    llvm::APInt genAPIntInstance(int numElements);

    llvm::APFloat genAPFloatInstance(int numElements);

    llvm::Value* codeGen(Utils::IRContext* context) override;
    void semanticPass(Utils::IRContext* context) override;
    void dimensionPass(Analysis::DimensionSymbolTable* nt) override;
    [[nodiscard]] std::string toTree(const std::string& prefix, const std::string& childPrefix) const override;
    [[nodiscard]] bool isConst() const noexcept override;
    [[nodiscard]] std::vector<std::shared_ptr<AST::ExprNode>> constData(
        std::shared_ptr<AST::ExprNode>& me) const override;
};
}  // namespace AST
