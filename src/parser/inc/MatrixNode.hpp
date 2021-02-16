#pragma once

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>

#include <iostream>
#include <vector>

#include "ExprASTNode.hpp"
#include "LiteralNode.hpp"
#include "Type.hpp"

namespace AST {
class MatrixNode : public ExprNode {
   public:
    std::vector<std::shared_ptr<ExprNode>> data;

    std::vector<uint> getDimensions();
    llvm::APInt genAPIntInstance(int numElements);

    llvm::APFloat genAPFloatInstance(int numElements);

    llvm::Value* codeGen(Utils::IRContext* context) override;
    void semanticPass() override;
};
}  // namespace AST
