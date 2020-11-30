#pragma once

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>

#include <vector>

#include "ExprASTNode.hpp"
#include "LiteralNode.hpp"
#include "Type.hpp"

namespace Analysis {
class NameTable;
}

namespace AST {
class MatrixNode : public ExprNode {
   public:
    std::vector<std::vector<std::shared_ptr<ExprNode>>> data;

    std::vector<int> getDimensions();
    llvm::APInt genAPIntInstance(int numElements);
    llvm::APFloat genAPFloatInstance(int numElements);
    llvm::Value* codeGen(Utils::IRContext* context) override;
    void dimensionPass(Analysis::NameTable* nt) override;
};
}  // namespace AST
