#pragma once

#include <llvm-10/llvm/ADT/APFloat.h>
#include <llvm-10/llvm/ADT/APInt.h>

#include <vector>

#include "ExprASTNode.hpp"
#include "LiteralASTNode.hpp"
#include "Type.hpp"

namespace AST {
class MatrixASTNode : public ExprAST {
   public:
    std::vector<std::vector<std::shared_ptr<ExprAST>>> data;

    int numElements();
    llvm::APInt genAPIntInstance(int numElements);
    llvm::APFloat genAPFloatInstance(int numElements);

    llvm::Value* codeGen(llvm::Module* module, llvm::IRBuilder<>* Builder,
                         llvm::Function* fp) override;
    void dimensionPass() override;
};
}  // namespace AST
