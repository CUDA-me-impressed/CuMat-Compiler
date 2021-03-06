#include "MatrixNode.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Type.h>

#include <iostream>

#include "CodeGenUtils.hpp"
#include "TypeCheckingUtils.hpp"

llvm::Value* AST::MatrixNode::codeGen(Utils::IRContext* context) {
    //    // Get the LLVM type out for the basic type
    //    Typing::MatrixType matTypeAST = std::get<Typing::MatrixType>(*this->type);
    //    llvm::Type* ty = matTypeAST.getLLVMType(context);
    //    // Get function to store this data within
    //    llvm::ArrayType* matType = llvm::ArrayType::get(ty, matTypeAST.getLength());
    //
    //    // Create a store instance for the correct precision and data type
    //    // Address space set to zero
    //    auto matAlloc = context->Builder->CreateAlloca(matType, 0, nullptr, "matVar");
    //
    //    // We need to fill in the data for each of the elements of the array:
    //    std::vector<llvm::Value*> matElements(matTypeAST.getLength());
    //    auto zero = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, 0, true));
    //    for (int row = 0; row < data.size(); row++) {
    //        for (int column = 0; column < data[row].size(); column++) {
    //            // Generate the code for the element -> The Value* will be what
    //            // we store within the matrix location so depending on what we are
    //            // storing, it must be sufficient to run
    //            size_t elIndex = row * data.size() + column;
    //            llvm::Value* val = data[row][column]->codeGen(context);
    //
    //            // Create index for current index of the value
    //            auto index = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, elIndex, true));
    //
    //            // Get pointer to the index location within memory
    //            auto ptr = llvm::GetElementPtrInst::Create(matType, matAlloc, {zero, index}, "",
    //                                                       context->Builder->GetInsertBlock());
    //            context->Builder->CreateStore(val, ptr);
    //        }
    //    }
    //    return matAlloc;
    // Assume every expr is an evaluated matrix
    auto matAlloc = Utils::createMatrix(context, *this->type);
    auto matRecord = Utils::getMatrixFromPointer(context, matAlloc);

    // Compress literal nodes to matrix representation
    for (int i = 0; i < this->data.size(); i++) {
        auto literal = this->data.at(i);
        llvm::Value* literalLLVMMat = literal->codeGen(context);
        auto literalRecord = Utils::getMatrixFromPointer(context, literalLLVMMat);
        // Get the value at the pointer
        llvm::Value* literalLLVVal = Utils::getValueFromPointerOffset(context, literalRecord.dataPtr, 0, "literalVal");
        Utils::insertValueAtPointerOffset(context, matRecord.dataPtr, 0, literalLLVVal);

        // Clean up the allocation for the literal, else memory leak big time!
        auto* freeMat = llvm::CallInst::CreateFree(literalLLVMMat, context->Builder->GetInsertBlock());
        context->Builder->Insert(freeMat, "matFree");
    }

    return matAlloc;
}

llvm::APInt AST::MatrixNode::genAPIntInstance(const int numElements) {
    if (std::get<Typing::MatrixType>(*(this->type)).primType == Typing::PRIMITIVE::INT ||
        std::get<Typing::MatrixType>(*(this->type)).primType == Typing::PRIMITIVE::BOOL) {
        return llvm::APInt(std::get<Typing::MatrixType>(*(this->type)).offset(), numElements);
    }
    std::cerr << "Attempting to assign arbitrary precision integer type"
              << " to internal non-integer type [" << this->literalText << "]" << std::endl;
    return llvm::APInt();
}

/**
 * Returns a list of vectors with the size of each dimension or indicates if
 * the dimension is dynamically sized
 * @return
 */
std::vector<uint> AST::MatrixNode::getDimensions() {
    auto matType = std::get_if<Typing::MatrixType>(&*type);
    return matType ? matType->dimensions : std::vector<uint>();
}

void AST::MatrixNode::semanticPass(Utils::IRContext* context) {
    //    std::cout << "Performing Matrix Semantic Pass" << std::endl;
    for (auto const& child : this->children) child->semanticPass(context);
    bool sameType = true;
    bool zeroRank = true;
    Typing::PRIMITIVE primType = Typing::PRIMITIVE::NONE;
    Typing::MatrixType exprType;
    // Runs through all elements to check types and ranks
    for (auto& el : data) {
        try {
            exprType = std::get<Typing::MatrixType>(*el->type);
        } catch (std::bad_cast b) {
            std::cout << "Caught: " << b.what();
        }
        Typing::PRIMITIVE prim = exprType.getPrimitiveType();
        if (primType == Typing::PRIMITIVE::NONE && prim != Typing::PRIMITIVE::NONE) {
            primType = prim;
        } else {
            sameType = sameType && (primType == prim);
        }
        zeroRank = zeroRank && (exprType.rank == 0);
        if (!(sameType)) {
            std::cout << "All elements of matrix must be of same type";
            std::exit(2);
        } else if (!(zeroRank)) {
            std::cout << "All elements of matrix must be scalars";
            std::exit(2);
        }
    }

    std::vector<uint> dimensions = this->getDimensions();

    this->type = TypeCheckUtils::makeMatrixType(dimensions, primType);
}
std::string AST::MatrixNode::toTree(const std::string& prefix, const std::string& childPrefix) const {
    return ExprNode::toTree(prefix, childPrefix);
}
