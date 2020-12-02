#include "CodeGenUtils.hpp"

#include <algorithm>
#include <iostream>

llvm::AllocaInst* Utils::createMatrix(Utils::IRContext* context, const Typing::Type& type) {
    // We need a prefix that has some basic information
    Typing::MatrixType matType;
    try {
        matType = std::get<Typing::MatrixType>(type);
    } catch (std::bad_variant_access& bva) {
        std::cerr << "Failed to allocate matrix type!" << bva.what() << std::endl;
    }
    int matLength = 0;
    // TODO: Dynamic size
    matLength = matType.getLength();

    // Generate actual array with offset for dimension information
    llvm::Type* ty = matType.getLLVMType(context->module);

    //
    llvm::ArrayType* matHeaderType = llvm::ArrayType::get(ty, matType.rank + 3);
    llvm::ArrayType* matDataType = llvm::ArrayType::get(ty, matLength);

    llvm::ConstantInt* matSizeLLVM =
        llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, matLength, true));
    llvm::ConstantInt* matSizeHeader =
        llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, matType.rank + 3, true));

    // Allocation pointers for header and end
    auto matAlloc = context->Builder->CreateAlloca(ty, 0, matSizeLLVM, "matVar");
    auto matHeaderAlloc = context->Builder->CreateAlloca(llvm::Type::getInt64Ty(context->module->getContext()), 0,
                                                         matSizeHeader, "matHeader");

    auto rank = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, matType.rank));
    auto numBytes = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, (matType.getLength() * matType.offset())/8));

    // Insert header into data
    insertRelativeToPointer(context, matHeaderType, matHeaderAlloc, 0, matAlloc); // For the pointer to the data
    insertRelativeToPointer(context, matHeaderType, matHeaderAlloc, 1, rank);     // For rank of matrix
    insertRelativeToPointer(context, matHeaderType, matHeaderAlloc,  2, numBytes);// For number of bytes in matrix

    // For the matrix dimensionality
    for (int i = 0; i <= matType.rank; i++) {
        auto val = llvm::ConstantInt::get(context->module->getContext(),
                                          llvm::APInt(64, matType.dimensions.at(i)));
        // Offset of 3 from before
        insertRelativeToPointer(context, matHeaderType, matHeaderAlloc, i+3, val);
    }
    return matHeaderAlloc;
}

std::unique_ptr<Utils::LLVMMatrixRecord> Utils::getMatrixFromPointer(IRContext* context, llvm::AllocaInst* basePtr) {
    // Let us get out some base info -> All addresses in memory are 64 bit
    auto dataPtrType = llvm::Type::getInt64PtrTy(context->module->getContext());
    auto headerType = llvm::Type::getInt64Ty(context->module->getContext());

    llvm::Value* dataPtr = Utils::getValueRelativeToPointer(context, dataPtrType, basePtr, 0);
    llvm::Value* rank = Utils::getValueRelativeToPointer(context, headerType, basePtr, 1);
    llvm::Value* numBytes = Utils::getValueRelativeToPointer(context, headerType, basePtr, 2);
    auto record = std::make_unique<Utils::LLVMMatrixRecord>();
    // Yes this is awful, no i dont care, yes the compiler will optimise it (I hope)
    record->dataPtr = dataPtr;
    record->rank = rank;
    record->numBytes = numBytes;
    return record;
}

/**
 * Inserts a LLVM Value relative to a pointer and an offset in number of elements
 * @param context
 * @param type
 * @param ptr - Base pointer
 * @param offset - Number of elements away (valsize in bits)
 * @param val - Value we are storing
 * @param valSize - Default 64 (8 Bytes)
 */
void Utils::insertRelativeToPointer(IRContext* context, llvm::Type* type, llvm::Value* ptr, int offset,
                                    llvm::Value* val) {
    auto zero = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(type->getPrimitiveSizeInBits(), 0, true));
    auto offsetIndex = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(type->getPrimitiveSizeInBits(), offset));
    auto offsetPtr =
        llvm::GetElementPtrInst::Create(type, ptr, {zero, offsetIndex}, "", context->Builder->GetInsertBlock());
    context->Builder->CreateStore(val, offsetPtr);
}

llvm::Value * Utils::getValueRelativeToPointer(IRContext* context, llvm::Type* type, llvm::Value* ptr, int offset) {
    auto zero = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(type->getPrimitiveSizeInBits(), 0, true));
    auto offsetIndex = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(type->getPrimitiveSizeInBits(), offset));
    auto ptrOffset = llvm::GetElementPtrInst::Create(type, ptr, {zero, offsetIndex}, "", context->Builder->GetInsertBlock());
    return context->Builder->CreateLoad(type, ptrOffset, "");
}