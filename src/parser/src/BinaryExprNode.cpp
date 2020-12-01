#include "BinaryExprNode.hpp"

#include <CodeGenUtils.hpp>
#include <MatrixNode.hpp>

static llvm::AllocaInst* CreateEntryBlockAlloca(llvm::IRBuilder<>& Builder,
                                                const std::string& VarName,
                                                llvm::Type* Type) {
    llvm::IRBuilder<> TmpB(
        &Builder.GetInsertBlock()->getParent()->getEntryBlock(),
        Builder.GetInsertBlock()->getParent()->getEntryBlock().begin());
    return TmpB.CreateAlloca(Type, nullptr, VarName);
}

llvm::Value* AST::BinaryExprNode::codeGen(Utils::IRContext* context) {
    // Assumption is that our types are two evaluated matricies of compatible
    // dimensions. We first generate code for each of the l and r matricies
    llvm::Value* lhsVal = lhs->codeGen(context);
    llvm::Value* rhsVal = rhs->codeGen(context);
    auto lhsMatType = std::dynamic_pointer_cast<AST::MatrixNode>(this->lhs);
    auto rhsMatType = std::dynamic_pointer_cast<AST::MatrixNode>(this->rhs);

    llvm::Type* lhsTy = lhsMatType->getLLVMType(context->module);
    llvm::Type* rhsTy = rhsMatType->getLLVMType(context->module);

    auto lhsDimension = lhsMatType->getDimensions();
    auto rhsDimension = rhsMatType->getDimensions();

    std::vector<int>* targetDimension{};
    if (lhsDimension.size() > rhsDimension.size()) {
        targetDimension = &lhsDimension;
    } else {
        targetDimension = &rhsDimension;
    }

    auto newMatAlloc = Utils::createMatrix(context, resultType);

    switch (op) {
        case PLUS: {
            plusCodeGen(context, lhsVal, rhsVal, lhsTy, rhsTy,
                        newMatAlloc);
            break;
        }
        default:
            // TODO: Remove when ALL functions are implemented
            break;
    }

    return nullptr;
}

void AST::BinaryExprNode::plusCodeGen(Utils::IRContext* context,
                                      llvm::Value* lhsVal, llvm::Value* rhsVal,
                                      const Typing::Type& lhsType,
                                      const Typing::Type& rhsType,
                                      llvm::AllocaInst* matAlloc) {
    auto Builder = context->Builder;
    llvm::Function* parent = Builder->GetInsertBlock()->getParent();

    llvm::BasicBlock* whileBB =
        llvm::BasicBlock::Create(Builder->getContext(), "add.loop", parent);
    llvm::BasicBlock* addBB =
        llvm::BasicBlock::Create(Builder->getContext(), "add.add", parent);
    llvm::BasicBlock* endBB =
        llvm::BasicBlock::Create(Builder->getContext(), "add.end", parent);

    auto indexAlloca = CreateEntryBlockAlloca(
        *Builder, "", llvm::Type::getInt64Ty(Builder->getContext()));

    Builder->CreateBr(whileBB);

    Builder->SetInsertPoint(whileBB);
    {
        auto* ind = Builder->CreateLoad(indexAlloca, "loadCounter");
        auto* val = Builder->CreateAdd(
            ind, llvm::ConstantInt::get(
                     llvm::Type::getInt64Ty(Builder->getContext()),
                     llvm::APInt{64, 1, false}));
        Builder->CreateStore(val, indexAlloca);
    }

    Builder->SetInsertPoint(addBB);
    {
        auto index = Builder->CreateLoad(indexAlloca);
        auto li = Builder->CreateURem();
        auto l = Builder->CreateExtractElement();
    }

    //    llvm::ArrayType* matType;
    //    if (dimension.size() == 1) {
    //        matType = llvm::ArrayType::get(lhsType, index * dimension.at(0));
    //    }
    //
    //    for (int i = 0; i < dimension.at(0); i++) {
    //        if (dimension.size() > 1) {
    //            // Create a new dimension vector with this dimension removed
    //            std::vector<int> subDimension(dimension.begin() + 1,
    //                                          dimension.end());
    //            plusCodeGen(TheModule, Builder, lhs, rhs, lhsType, rhsType,
    //                        matAlloc, subDimension, (index * prevDim) + i,
    //                        dimension.at(0));
    //        } else {
    //            // TODO: Make work with non-64 bit variables
    //            auto zero = llvm::ConstantInt::get(TheModule->getContext(),
    //                                               llvm::APInt(64, 0, true));
    //            auto indexVal = llvm::ConstantInt::get(
    //                TheModule->getContext(), llvm::APInt(64, index, true));
    //            // Pointer to the index within IR
    //
    //            auto ptrLhs = Builder->CreateExtractElement()
    //
    //            auto ptrLhs = llvm::GetElementPtrInst::Create(
    //                matType, lhs, {zero, indexVal}, "lhs",
    //                Builder->GetInsertBlock());
    //            auto ptrRhs = llvm::GetElementPtrInst::Create(
    //                matType, rhs, {zero, indexVal}, "rhs",
    //                Builder->GetInsertBlock());
    //            auto ptrNew = llvm::GetElementPtrInst::Create(
    //                matType, matAlloc, {zero, indexVal}, "",
    //                Builder->GetInsertBlock());
    //            // Compute the Addition
    //            auto addSum = Builder->CreateAdd(Builder->CreateLoad(ptrLhs),
    //                                             Builder->CreateLoad(ptrRhs));
    //            // Store the element at the correct position
    //            Builder->CreateStore(addSum, ptrNew);
    //        }
    //    }
}
