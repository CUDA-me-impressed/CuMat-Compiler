#include "LiteralNode.hpp"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Type.h>

#include <iostream>

/**
 * Returns an LLVM Constant type which we use to populate data entires
 * Literals consist of the lowest form of data structure we can have within
 * CuMat and as such this section details the creation of these within memory
 * @tparam T
 * @param module
 * @param Builder
 * @param fp
 * @return
 */
/*template <class T>
llvm::Value* AST::LiteralNode<T>::codeGen(Utils::IRContext* context) {
    llvm::Type* ty;
    switch (this->type->primType) {
        case Typing::PRIMITIVE::INT: {
            ty = static_cast<llvm::Type*>(
                llvm::Type::getInt64Ty(module->getContext()));
            return llvm::ConstantInt::get(ty, llvm::APInt(64, value, true));
        }
        case Typing::PRIMITIVE::FLOAT: {
            ty = llvm::Type::getFloatTy(module->getContext());
            return llvm::ConstantFP::get(ty, llvm::APFloat(value));
        }
        case Typing::PRIMITIVE::BOOL: {
            ty = static_cast<llvm::Type*>(
                llvm::Type::getInt1Ty(module->getContext()));
            return llvm::ConstantInt::get(ty, llvm::APInt(1, value, false));
        }
        default: {
            std::cerr << "Cannot find a valid type for " << this->literalText
                      << std::endl;
            // Assign the type to be an integer
            ty = static_cast<llvm::Type*>(
                llvm::Type::getInt64Ty(module->getContext()));
            break;
        }
        case Typing::PRIMITIVE::STRING:
            break;
        case Typing::PRIMITIVE::NONE:
            break;
    }
    return nullptr;
}*/

template <>
llvm::Value* AST::LiteralNode<int>::codeGen(Utils::IRContext* context) {
    if (std::get<Typing::MatrixType>(*type).primType != Typing::PRIMITIVE::INT) return nullptr;

    Typing::MatrixType matType{0, std::vector<uint>(), Typing::PRIMITIVE::INT};
    auto* matAlloc = Utils::createMatrix(context, matType);
    auto* dataPtr = Utils::getValueFromPointerOffset(context, matAlloc, 0, "matArrPtr");
    auto ty = static_cast<llvm::Type*>(llvm::Type::getInt64Ty(context->module->getContext()));
    auto* val = llvm::ConstantInt::get(ty, llvm::APInt(64, value, true));
    Utils::insertValueAtPointerOffset(context, dataPtr, 0, val);
    return matAlloc;
}

template <>
llvm::Value* AST::LiteralNode<float>::codeGen(Utils::IRContext* context) {
    if (std::get<Typing::MatrixType>(*type).primType != Typing::PRIMITIVE::FLOAT) return nullptr;
    auto ty = llvm::Type::getFloatTy(context->module->getContext());
    return llvm::ConstantFP::get(ty, llvm::APFloat(value));
}

template <>
llvm::Value* AST::LiteralNode<std::string>::codeGen(Utils::IRContext* context) {
    llvm::Type* ty;
    return nullptr;
}

// Yes, I could have template meta-programmed this to be general, but that's nasty
template <>
std::string AST::LiteralNode<float>::toTree(const std::string& prefix, const std::string& childPrefix) const {
    return prefix + std::to_string(value) + '\n';
}
template <>
std::string AST::LiteralNode<int>::toTree(const std::string& prefix, const std::string& childPrefix) const {
    return prefix + std::to_string(value) + '\n';
}
template <>
std::string AST::LiteralNode<std::string>::toTree(const std::string& prefix, const std::string& childPrefix) const {
    return prefix + value + '\n';
}

template class AST::LiteralNode<float>;
template class AST::LiteralNode<int>;
template class AST::LiteralNode<std::string>;