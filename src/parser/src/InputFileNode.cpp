#include "InputFileNode.hpp"

#include <iostream>
#include <vector>

#include "CodeGenUtils.hpp"
#include "DimensionPass.hpp"
#include "DimensionsSymbolTable.hpp"
#include "TreePrint.hpp"

llvm::Value* AST::InputFileNode::codeGen(Utils::IRContext* context) {
    //TODO: CALL C++ FUNCTION FOR THIS

    if(auto retType = std::get_if<Typing::MatrixType>(this->type.get()))
    {
        auto rank = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, retType->rank));

        llvm::Type* i32type = llvm::Type::getInt32Ty(context->module->getContext());
        // For the matrix dimensionality
        llvm::ArrayType* matDimensionType = llvm::ArrayType::get(llvm::Type::getInt64Ty(context->module->getContext()), retType->rank);
        // Allocation of the matrix data
        llvm::Constant* matDimAllocaSize = llvm::ConstantExpr::getSizeOf(matDimensionType);
        // This will by default be i64, need to cast to i32 (I think its safe)
        //matDimAllocaSize = llvm::ConstantExpr::getTruncOrBitCast(matDimAllocaSize, i32type);
        auto* matDimAlloc = context->Builder->CreateAlloca(matDimensionType, matDimAllocaSize,"fileReadDimensions");
        //context->Builder->Insert(matDimAlloc, "matDimData");

        for (int i = 0; i < retType->rank; i++) {
            auto val = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, retType->dimensions.at(i)));

            // Offset of 3 from before
            insertValueAtPointerOffset(context, matDimAlloc, i, val, false);
        }

        //Strings (Good luck me!)
        llvm::Type* i8type = llvm::Type::getInt8Ty(context->module->getContext());
        llvm::ArrayType* stringArrayType = llvm::ArrayType::get(i8type, this->fileName.size());
        llvm::Constant* stringAllocaSize = llvm::ConstantExpr::getSizeOf(stringArrayType);
        //stringAllocaSize = llvm::ConstantExpr::getTruncOrBitCast(stringAllocaSize,i32type);
        auto* stringAlloc = context->Builder->CreateAlloca(stringArrayType, stringAllocaSize,"fileReadStringLiteral");
        //context->Builder->Insert(stringAlloc, "stringData");
        std::vector<llvm::Constant*> filenameVec;
        for(int i = 0; i < this->fileName.length(); i++)
        {
            auto val = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(8,this->fileName[i]));
            filenameVec.push_back(val);
//            insertValueAtPointerOffset(context, stringAlloc, i, val, false);
        }
        llvm::ArrayType* arrayType = llvm::ArrayType::get(i8type, filenameVec.size());
        llvm::Constant* init = llvm::ConstantArray::get(arrayType, filenameVec);
        context->Builder->CreateStore(init, stringAlloc);


        //Null terminator
        auto val = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(8,0));
        insertValueAtPointerOffset(context, stringAlloc, this->fileName.length(), val, false);

        auto* expectedDimType = llvm::ArrayType::get(llvm::Type::getInt64Ty(context->module->getContext()), 0)->getPointerTo();
        auto* expectedStringType = llvm::ArrayType::get(i8type, 0)->getPointerTo();

        auto newStringAlloc = context->Builder->CreateBitCast(stringAlloc,expectedStringType);
        auto newDimAlloc = context->Builder->CreateBitCast(matDimAlloc,expectedDimType);

        std::vector<llvm::Value*> argVals{newStringAlloc,newDimAlloc,rank};

        if (retType->primType == Typing::PRIMITIVE::INT) {
            return context->Builder->CreateCall(context->symbolTable->inputFunctions.funcInt, argVals);
        } else if (retType->primType == Typing::PRIMITIVE::FLOAT) {
            return context->Builder->CreateCall(context->symbolTable->inputFunctions.funcFloat, argVals);
        } else {
            throw std::runtime_error("Input File Node has a non-int/float primitive type");
        }
    } else
    {
        throw std::runtime_error("Type of input file node is not a matrix type. Something has gone horribly wrong");
    }
}

void AST::InputFileNode::semanticPass(Utils::IRContext* context) {
    //pass
}

std::string AST::InputFileNode::toTree(const std::string& prefix, const std::string& childPrefix) const {
    return prefix + "Input From File";
}


void AST::InputFileNode::dimensionPass(Analysis::DimensionSymbolTable* nt) {
    auto* type = std::get_if<Typing::MatrixType>(this->type.get());
    if (type) {
        for (uint i : type->dimensions) {
            if (i == 0) {
                dimension_error("Invalid dimensional argument to input function. Must be greater than 0", this);
            }
        }
    }
}
