#include "BinaryExprNode.hpp"

#include <CodeGenUtils.hpp>
#include <MatrixNode.hpp>

llvm::Value* AST::BinaryExprNode::codeGen(llvm::Module* module,
                                          llvm::IRBuilder<>* Builder,
                                          llvm::Function* fp) {
    // Assumption is that our types are two evaluated matricies of compatible
    // dimensions. We first generate code for each of the l and r matricies
    llvm::Value* lhsVal = lhs->codeGen(module, Builder, fp);
    llvm::Value* rhsVal = rhs->codeGen(module, Builder, fp);
    auto lhsMatType = std::dynamic_pointer_cast<AST::MatrixNode>(this->lhs);
    auto rhsMatType = std::dynamic_pointer_cast<AST::MatrixNode>(this->rhs);

    llvm::Type* lhsTy = lhsMatType->getLLVMType(module);
    llvm::Type* rhsTy = rhsMatType->getLLVMType(module);

    auto lhsDimension = lhsMatType->getDimensions();
    auto rhsDimension = rhsMatType->getDimensions();

    auto newMatAlloc = Utils::generateMatrixAllocation(lhsTy, lhsDimension, Builder);


    switch (op) {
        case PLUS: {
            plusCodeGen(module, Builder, lhsVal, rhsVal, lhsTy, rhsTy, newMatAlloc, lhsDimension);
            break;
        }
        default:
            // TODO: Remove when ALL functions are implemented
            break;
    }

    return nullptr;
}

void AST::BinaryExprNode::plusCodeGen(
    llvm::Module* TheModule, llvm::IRBuilder<>* Builder,
    llvm::Value* lhs, llvm::Value* rhs,
    llvm::Type* lhsType, llvm::Type* rhsType,
    llvm::AllocaInst* matAlloc,
    std::vector<int> dimension, int index, int prevDim) {
    llvm::ArrayType* matType;
    if (dimension.size() == 1) {
        matType = llvm::ArrayType::get(lhsType, index * dimension.at(0));
    }

    for (int i = 0; i < dimension.at(0); i++) {
        if (dimension.size() > 1) {
            // Create a new dimension vector with this dimension removed
            std::vector<int> subDimension(dimension.begin() + 1,
                                          dimension.end());
            plusCodeGen(TheModule, Builder, lhs, rhs, lhsType, rhsType, matAlloc,
                        subDimension, (index * prevDim) + i, dimension.at(0));
        } else {
            // TODO: Make work with non-64 bit variables
            auto zero = llvm::ConstantInt::get(TheModule->getContext(),
                                               llvm::APInt(64, 0, true));
            auto indexVal = llvm::ConstantInt::get(
                TheModule->getContext(), llvm::APInt(64, index, true));
            // Pointer to the index within IR
            auto ptrLhs = llvm::GetElementPtrInst::Create(
                matType, lhs, {zero, indexVal}, "lhs",
                Builder->GetInsertBlock());
            auto ptrRhs = llvm::GetElementPtrInst::Create(
                matType, rhs, {zero, indexVal}, "rhs",
                Builder->GetInsertBlock());
            auto ptrNew = llvm::GetElementPtrInst::Create(
                matType, matAlloc, {zero, indexVal}, "", Builder->GetInsertBlock());
            // Compute the Addition
            auto addSum = Builder->CreateAdd(Builder->CreateLoad(ptrLhs),
                                             Builder->CreateLoad(ptrRhs));
            // Store the element at the correct position
            Builder->CreateStore(addSum, ptrNew);
        }
    }
}
