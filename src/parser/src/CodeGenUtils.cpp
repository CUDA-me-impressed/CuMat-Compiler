#include "CodeGenUtils.hpp"

#include <algorithm>

llvm::AllocaInst* Utils::MatrixInterface::createMatrix(
    Typing::Type type, llvm::IRBuilder<>* Builder) {
    // We need a prefix that has some basic information
    int numDimensions = type->getNumDimensions();

    // Generate actual array with offset for dimension information
    llvm::Type* ty = type->getLLVMType();
    // Num Dimensions + 1 as we want to store dimensionality information plus initial
    llvm::ArrayType* matType = llvm::ArrayType::get(ty, type->numElements()+numDimensions+1);
    auto matAlloc = Builder->CreateAlloca(ty, 0, type->numElements()+numDimensions+1, "matVar");

    // Generate prefix
    llvm::Value* rank = llvm::ConstantInt();
    for(int i = 0; i < numDimensions; i++){
    }
}
