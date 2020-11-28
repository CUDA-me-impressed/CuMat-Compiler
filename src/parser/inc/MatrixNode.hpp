#pragma once

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>

#include <vector>

#include "ExprASTNode.hpp"
#include "LiteralNode.hpp"
#include "Type.hpp"

namespace AST {
class MatrixNode : public ExprAST {
   public:
    std::vector<std::vector<std::shared_ptr<ExprAST>>> data;

    int numElements();
    std::vector<int> getDimensions();
    llvm::APInt genAPIntInstance(int numElements);
    llvm::APFloat genAPFloatInstance(int numElements);
    llvm::Value* codeGen(llvm::Module* module, llvm::IRBuilder<>* Builder,
                         llvm::Function* fp) override;
};
}  // namespace AST
