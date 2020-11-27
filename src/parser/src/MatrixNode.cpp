#include "MatrixNode.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Type.h>

#include <iostream>

#include "CodeGenUtils.hpp"

llvm::Value* AST::MatrixNode::codeGen(llvm::Module* module,
                                      llvm::IRBuilder<>* Builder,
                                      llvm::Function* fp) {
    llvm::Type* ty;
    switch (this->type->primType) {
        case Typing::PRIMITIVE::INT: {
            ty = static_cast<llvm::Type*>(
                llvm::Type::getInt64Ty(module->getContext()));
            break;
        }
        case Typing::PRIMITIVE::FLOAT: {
            ty = llvm::Type::getFloatTy(module->getContext());
            break;
        }
        case Typing::PRIMITIVE::BOOL: {
            ty = static_cast<llvm::Type*>(
                llvm::Type::getInt1Ty(module->getContext()));
            break;
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
    // Get function to store this data within
    llvm::ArrayType* matType = llvm::ArrayType::get(ty, this->numElements());

    // Create a store instance for the correct precision and data type
    // Address space set to zero
    llvm::AllocaInst* matAlloc =
        new llvm::AllocaInst(ty, 0, "matVar", Builder->GetInsertBlock());

    // We need to fill in the data for each of the elements of the array:
    std::vector<llvm::Value*> matElements(this->numElements());
    auto zero =
        llvm::ConstantInt::get(module->getContext(), llvm::APInt(64, 0, true));
    for (int row = 0; row < data.size(); row++) {
        for (int column = 0; column < data[row].size(); column++) {
            // Generate the code for the element -> The Value* will be what
            // we store within the matrix location so depending on what we are
            // storing, it must be sufficient to run
            int elIndex = row * data.size() + column;
            llvm::Value* val = data[row][column]->codeGen(module, Builder, fp);

            // Create index for current index of the value
            auto index = llvm::ConstantInt::get(module->getContext(),
                                                llvm::APInt(64, elIndex, true));

            // Get pointer to the index location within memory
            auto ptr = llvm::GetElementPtrInst::Create(
                matType, matAlloc, {zero, index}, "",
                Builder->GetInsertBlock());
            auto store = new llvm::StoreInst(index, ptr, false,
                                             Builder->GetInsertBlock());
        }
    }
    Utils::AllocSymbolTable[this->literalText] = matAlloc;
    return matAlloc;
}

int AST::MatrixNode::numElements() {
    // We assume all sides equal lengths
    return this->data.size() * this->data.at(0).size();
}
llvm::APInt AST::MatrixNode::genAPIntInstance(const int numElements) {
    if (this->type->primType == Typing::PRIMITIVE::INT ||
        this->type->primType == Typing::PRIMITIVE::BOOL) {
        return llvm::APInt(this->type->offset(), numElements);
    }
    std::cerr << "Attempting to assign arbitrary precision integer type"
              << " to internal non-integer type [" << this->literalText << "]"
              << std::endl;
    return llvm::APInt();
}

// llvm::APFloat AST::MatrixNode::genAPFloatInstance(const int numElements) {
//    //    if (this->type->primType == Typing::PRIMITIVE::FLOAT) {
//    //        return llvm::APFloat(64, numElements);
//    //    }
//    //    std::cerr << "Attempting to assign arbitrary precision float type"
//    //              << " to internal non-float type [" << this->literalText <<
//    //              "]"
//    //              << std::endl;
//    // TODO: Fix this floating points so they work
//    return llvm::APFloat(5.0f);
//}
