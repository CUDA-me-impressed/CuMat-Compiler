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
    // Get the LLVM type out for the basic type
    Typing::MatrixType matTypeAST = std::get<Typing::MatrixType>(*this->type);
    llvm::Type* ty = matTypeAST.getLLVMType(context);
    // Get function to store this data within
    llvm::ArrayType* matType = llvm::ArrayType::get(ty, matTypeAST.getLength());

    // Create a store instance for the correct precision and data type
    // Address space set to zero
    auto matAlloc = context->Builder->CreateAlloca(matType, 0, nullptr, "matVar");

    // We need to fill in the data for each of the elements of the array:
    std::vector<llvm::Value*> matElements(matTypeAST.getLength());
    auto zero = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, 0, true));
    for (int row = 0; row < data.size(); row++) {
        for (int column = 0; column < data[row].size(); column++) {
            // Generate the code for the element -> The Value* will be what
            // we store within the matrix location so depending on what we are
            // storing, it must be sufficient to run
            size_t elIndex = row * data.size() + column;
            llvm::Value* val = data[row][column]->codeGen(context);

            // Create index for current index of the value
            auto index = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, elIndex, true));

            // Get pointer to the index location within memory
            auto ptr = llvm::GetElementPtrInst::Create(matType, matAlloc, {zero, index}, "",
                                                       context->Builder->GetInsertBlock());
            context->Builder->CreateStore(val, ptr);
        }
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
    // TODO: Fix with Thomas's dimension change
    return std::vector<uint>({static_cast<uint>(data.size()), static_cast<uint>(data[0].size())});
}

void AST::MatrixNode::semanticPass() {
    //    std::cout << "Performing Matrix Semantic Pass" << std::endl;
    for (auto const& child : this->children) child->semanticPass();
    bool sameType = true;
    bool zeroRank = true;
    Typing::PRIMITIVE primType = Typing::PRIMITIVE::NONE;
    Typing::MatrixType exprType;
    // Runs through all elements to check types and ranks
    for (auto& row : data) {
        for (auto& el : row) {
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
    }

    std::vector<uint> dimensions = this->getDimensions();

    this->type = makeMatrixType(dimensions, primType);
}
