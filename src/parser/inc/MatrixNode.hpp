#pragma once

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>

#include <vector>

#include "ExprASTNode.hpp"
#include "LiteralNode.hpp"
#include "Type.hpp"

namespace AST {
class MatrixNode : public ExprNode {
   public:
    std::vector<std::vector<std::shared_ptr<ExprNode>>> data;

    size_t numElements();
    std::vector<int> getDimensions();
    llvm::APInt genAPIntInstance(int numElements);
    llvm::Type* getLLVMType(llvm::Module* module);
    llvm::APFloat genAPFloatInstance(int numElements);
    llvm::Value* codeGen(Utils::IRContext* context) override;
};
}  // namespace AST
