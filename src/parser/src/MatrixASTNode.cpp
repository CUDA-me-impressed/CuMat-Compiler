#include "MatrixASTNode.hpp"

#include <llvm-10/llvm/IR/DerivedTypes.h>
#include <llvm-10/llvm/IR/Type.h>

#include <iostream>

#include "LLVMUtils.hpp"

void AST::MatrixASTNode::codeGen() {
    llvm::Type* type;
    switch (this->type->primType) {
        case Typing::PRIMITIVE::INT: {
            type =
                llvm::IntegerType::getInt64Ty(LLVMUtils::module->getContext());
            break;
        }
        case Typing::PRIMITIVE::FLOAT: {
            type =
                llvm::IntegerType::getFloatTy(LLVMUtils::module->getContext());
            break;
        }
        case Typing::PRIMITIVE::BOOL: {
            type =
                llvm::IntegerType::getInt1Ty(LLVMUtils::module->getContext());
            break;
        }
        default: {
            std::cerr << "Cannot find a valid type for " << this->literalText
                      << std::endl;
            // Assign the type to be an integer
            type =
                llvm::IntegerType::getInt64Ty(LLVMUtils::module->getContext());
            break;
        }
    }
    llvm::ArrayType* array = llvm::ArrayType::get(type, this->numElements());
}

int AST::MatrixASTNode::numElements() {
    // We assume all sides equal lengths
    return this->data.size() * this->data.at(0).size();
}