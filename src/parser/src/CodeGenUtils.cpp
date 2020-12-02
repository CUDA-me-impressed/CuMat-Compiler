#include "CodeGenUtils.hpp"

#include <algorithm>

    llvm::AllocaInst* Utils::createMatrix(Utils::IRContext* context,
    const Typing::Type &type) {
    // We need a prefix that has some basic information
    Typing::MatrixType matType;
    try {
        matType = std::get<Typing::MatrixType>(type);
    } catch (std::bad_variant_access &bva) {
        std::cerr << "Failed to allocate matrix type!" << bva.what()
                  << std::endl;
    }
    int matLength = 0;
    // TODO: Dynamic size
    matLength = matType.getLength();

    // Generate actual array with offset for dimension information
    llvm::Type* ty = matType.getLLVMType(context->module);

    //
    llvm::ArrayType* matHeaderType = llvm::ArrayType::get(ty, matType.rank + 3);


    llvm::ConstantInt* matSizeLLVM = llvm::ConstantInt::get(context->module->getContext(),
                                                            llvm::APInt(64, matLength, true));
    llvm::ConstantInt* matSizeHeader = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, matType.rank+3, true));
    auto matAlloc = context->Builder->CreateAlloca(ty, 0, matSizeLLVM, "matVar");
    auto matHeaderAlloc = context->Builder->CreateAlloca(llvm::Type::getInt64Ty(context->module->getContext()),
0, matSizeHeader, "matHeader");

    // Generate prefix
    auto zero =
        llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, 0, true));
    auto rank = llvm::ConstantInt::get(context->module->getContext(),llvm::APInt(64, matType.rank));

    // For the pointer to the data
    auto index = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, 0));
    auto ptr = llvm::GetElementPtrInst::Create(matHeaderType, matAlloc, {zero, index}, "", context->Builder->GetInsertBlock());
    context->Builder->CreateStore(matAlloc, ptr);
        insertRelativeToPointer(context, matHeaderType, )

    // For the Rank

        for (int i = 0; i <= matType.rank; i++) {
        auto index =
            llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, i));
        auto ptr = llvm::GetElementPtrInst::Create(
            matRawType, matAlloc, {zero, index}, "", context->Builder->GetInsertBlock());
        llvm::Value* val =
            (i == 0) ? rank
                     : llvm::ConstantInt::get(
                           context->module->getContext(),
                           llvm::APInt(64, matType.dimensions.at(i - 1)));
        // Store the header information
        context->Builder->CreateStore(val, ptr);
    }
    return matAlloc;
}


void Utils::insertRelativeToPointer(IRContext* context, llvm::Type* type,
                                    llvm::Value* ptr, int offset, llvm::Value* val) {
    auto zero = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, 0, true));
    auto offsetIndex = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, offset));
    auto offsetPtr = llvm::GetElementPtrInst::Create(type, ptr, {zero, offsetIndex}, "", context->Builder->GetInsertBlock());
    context->Builder->CreateStore(val, offsetPtr);
}