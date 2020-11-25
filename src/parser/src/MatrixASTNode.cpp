#include "MatrixASTNode.hpp"

#include <llvm-10/llvm/IR/DerivedTypes.h>
#include <llvm-10/llvm/IR/Type.h>

#include <iostream>

void AST::MatrixASTNode::codeGen(llvm::Module* module) {
    llvm::Type* type;
    switch (this->type->primType) {
        case Typing::PRIMITIVE::INT: {
            type = static_cast<llvm::Type*>(
                llvm::Type::getInt64Ty(module->getContext()));
            break;
        }
        case Typing::PRIMITIVE::FLOAT: {
            type = llvm::Type::getFloatTy(module->getContext());
            break;
        }
        case Typing::PRIMITIVE::BOOL: {
            type = static_cast<llvm::Type*>(
                llvm::Type::getInt1Ty(module->getContext()));
            break;
        }
        default: {
            std::cerr << "Cannot find a valid type for " << this->literalText
                      << std::endl;
            // Assign the type to be an integer
            type = static_cast<llvm::Type*>(
                llvm::Type::getInt64Ty(module->getContext()));
            break;
        }
        case Typing::PRIMITIVE::STRING:
            break;
        case Typing::PRIMITIVE::NONE:
            break;
    }
    llvm::ArrayType* array = llvm::ArrayType::get(type, this->numElements());
}

int AST::MatrixASTNode::numElements() {
    // We assume all sides equal lengths
    return this->data.size() * this->data.at(0).size();
}