#include "LiteralNode.hpp"

#include <llvm-10/llvm/ADT/APFloat.h>
#include <llvm-10/llvm/ADT/APInt.h>
#include <llvm-10/llvm/IR/Constants.h>
#include <llvm-10/llvm/IR/Type.h>

#include <iostream>

/**
 * Returns an LLVM Constant type which we use to populate a matrix
 * @tparam T
 * @param module
 * @param Builder
 * @param fp
 * @return
 */
template <class T>
llvm::Value* AST::LiteralNode<T>::codeGen(llvm::Module* module,
                                             llvm::IRBuilder<>* Builder,
                                             llvm::Function* fp) {
    llvm::Type* ty;
    switch (this->type->primType) {
        case Typing::PRIMITIVE::INT: {
            type = static_cast<llvm::Type*>(
                llvm::Type::getInt64Ty(module->getContext()));
            return llvm::ConstantInt::get(ty, llvm::APInt(64, value, true));
        }
        case Typing::PRIMITIVE::FLOAT: {
            type = llvm::Type::getFloatTy(module->getContext());
            return llvm::ConstantFP::get(ty, llvm::APFloat(value));
        }
        case Typing::PRIMITIVE::BOOL: {
            type = static_cast<llvm::Type*>(
                llvm::Type::getInt1Ty(module->getContext()));
            return llvm::ConstantInt::get(ty, llvm::APInt(1, value, false));
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
    return nullptr;
}