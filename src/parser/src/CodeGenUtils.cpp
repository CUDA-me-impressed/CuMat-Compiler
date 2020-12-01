#include "CodeGenUtils.hpp"

#include <algorithm>
#include <iostream>

    llvm::AllocaInst* Utils::MatrixInterface::createMatrix(Utils::IRContext* context,
    Typing::Type type) {
    // We need a prefix that has some basic information
    Typing::MatrixType matType;
    try {
        matType = std::get<Typing::MatrixType>(type);
    } catch (std::bad_variant_access bva) {
        std::cerr << "Failed to allocate matrix type!" << bva.what()
                  << std::endl;
    }
    int matLength = 0;
    // TODO: Dynamic size
    matLength = matType.getLength();

    // Generate actual array with offset for dimension information
    llvm::Type* ty = matType.getLLVMType(context->module);
    // Num Dimensions + 1 as we want to store dimensionality information plus
    // initial
    llvm::ArrayType* matRawType =
        llvm::ArrayType::get(ty, matLength + matType.rank + 1);
    llvm::ConstantInt* matSizeLLVM = llvm::ConstantInt::get(
        context->module->getContext(), llvm::APInt(64, matLength, true));
    auto matAlloc = context->Builder->CreateAlloca(ty, 0, matSizeLLVM, "matVar");

    // Generate prefix
    auto zero =
        llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, 0, true));
    auto rank = llvm::ConstantInt::get(context->module->getContext(),
                                       llvm::APInt(64, matType.rank));
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
