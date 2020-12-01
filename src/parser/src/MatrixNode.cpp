#include "MatrixNode.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Type.h>

#include <iostream>
#include <numeric>

#include "CodeGenUtils.hpp"

llvm::Value* AST::MatrixNode::codeGen(Utils::IRContext* context) {
    // Get the LLVM type out for the basic type
    auto matType = std::get<Typing::MatrixType>(*type);
    auto matAlloc = Utils::createMatrix(context, matType);

    // TODO: Fill in data here

    Utils::AllocSymbolTable[this->literalText] = matAlloc;
    return matAlloc;
}