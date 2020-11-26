#include "MatrixNode.hpp"

#include <llvm-10/llvm/IR/Constants.h>
#include <llvm-10/llvm/IR/DerivedTypes.h>
#include <llvm-10/llvm/IR/Function.h>
#include <llvm-10/llvm/IR/IRBuilder.h>
#include <llvm-10/llvm/IR/Instructions.h>
#include <llvm-10/llvm/IR/Type.h>

#include <iostream>

#include "CodeGenUtils.hpp"

llvm::Value* AST::MatrixNode::codeGen(llvm::Module* module,
                                         llvm::IRBuilder<>* Builder,
                                         llvm::Function* fp) {
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
    llvm::ArrayType* mat_type = llvm::ArrayType::get(type, this->numElements());
    // We need a builder for the function
    llvm::IRBuilder<> tmpB(&fp->getEntryBlock(), fp->getEntryBlock().begin());
    llvm::AllocaInst* matAlloc =
        tmpB.CreateAlloca(mat_type, nullptr, this->literalText);
    // Create a store instance for the correct precision and data type
    llvm::StoreInst* storeInst;
    llvm::Value* val;
    if (this->type->primType == Typing::PRIMITIVE::INT ||
        this->type->primType == Typing::PRIMITIVE::BOOL) {
        val =
            llvm::ConstantInt::get(type, genAPIntInstance(this->numElements())),
        matAlloc;
    } else if (this->type->primType == Typing::PRIMITIVE::FLOAT) {
        //        storeInst = new llvm::StoreInst(
        //            llvm::ConstantFP(type,
        //            genAPFloatInstance(this->numElements())), matAlloc, fp);
    }
    Builder->CreateStore(val, matAlloc);
    Utils::AllocSymbolTable[this->literalText] = matAlloc;

    // We need to fill in the data for each of the elements of the array:
    for (int row = 0; row < data.size(); row++) {
        for (int column = 0; column < data[row].size(); column++) {
            // Generate the code for the element -> The Value* will be what
            // we store within the matrix location so depending on what we are
            // storing, it must be sufficient to run
            llvm::Value* elementValue =
                data[row][column]->codeGen(module, Builder, fp);
        }
    }

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

llvm::APFloat AST::MatrixNode::genAPFloatInstance(const int numElements) {
    //    if (this->type->primType == Typing::PRIMITIVE::FLOAT) {
    //        return llvm::APFloat(64, numElements);
    //    }
    //    std::cerr << "Attempting to assign arbitrary precision float type"
    //              << " to internal non-float type [" << this->literalText <<
    //              "]"
    //              << std::endl;
    // TODO: Fix this floating points so they work
    return llvm::APFloat(5.0f);
}
