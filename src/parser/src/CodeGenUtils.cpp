#include "CodeGenUtils.hpp"

#include <algorithm>
#include <iostream>

// TODO: USE NEW getLLVMType change to get the proper type
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
    llvm::Type* ty = matType.getLLVMPrimitiveType(context);
    auto* matHeaderType = matType.getLLVMType(context);

    llvm::ArrayType* matDataType = llvm::ArrayType::get(ty, matLength);
    llvm::ConstantInt* matSizeLLVM =
        llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, matLength, false));

    auto rank = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, matType.rank));
    auto numBytes = llvm::ConstantInt::get(context->module->getContext(),
                                           llvm::APInt(64, (matType.getLength() * matType.offset()) / 8));

    auto zeroOffset = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context->module->getContext()), 0);
    auto matAlloc = context->Builder->CreateAlloca(matDataType, matSizeLLVM, "matArr");
    auto matAllocPtr = context->Builder->CreateGEP(matAlloc, zeroOffset, "matArrPtr");

    auto matHeaderAlloc = context->Builder->CreateAlloca(matHeaderType, nullptr, "matStruct");

    insertValueAtPointerOffset(context, matHeaderAlloc, 0, matAllocPtr);
    insertValueAtPointerOffset(context, matHeaderAlloc, 1, rank);
    insertValueAtPointerOffset(context, matHeaderAlloc, 2, numBytes);

    // For the matrix dimensionality
    for (int i = 0; i < matType.rank; i++) {
        auto val = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, matType.dimensions.at(i)));

        // Offset of 3 from before
        insertValueAtPointerOffset(context, matHeaderAlloc, i + 3, val);
    }

    return matHeaderAlloc;
}

llvm::Value* Utils::getLength(IRContext* context, llvm::Value* basePtr, const Typing::MatrixType& type) {
    auto mat = Utils::getMatrixFromPointer(context, basePtr);
    auto value = context->Builder->CreateUDiv(
        mat.numBytes, llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, (type.offset()) / 8)));
    return value;
}

Utils::LLVMMatrixRecord Utils::getMatrixFromPointer(IRContext* context, llvm::Value* basePtr) {
    // Let us get out some base info -> All addresses in memory are 64 bit
    auto dataPtrType = llvm::Type::getInt64PtrTy(context->module->getContext());
    auto headerType = llvm::Type::getInt64Ty(context->module->getContext());

    llvm::Value* dataPtr = Utils::getValueRelativeToPointer(context, basePtr, 0, dataPtrType);
    llvm::Value* rank = Utils::getValueRelativeToPointer(context, basePtr, 1, headerType);
    llvm::Value* numBytes = Utils::getValueRelativeToPointer(context, basePtr, 2, headerType);
    return {dataPtr, rank, numBytes};
}

llvm::Value* Utils::getValueRelativeToPointer(IRContext* context, llvm::Value* ptr, llvm::Value* offsetIndex,
                                              llvm::Type* retType) {
    int size = 64;
    llvm::Type* type = ptr->getType();
    auto zero = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(size, 0, true));
    auto ptrOffset = context->Builder->CreateInBoundsGEP(ptr, {zero, offsetIndex}, "");
    return context->Builder->CreateLoad(retType, ptrOffset, "");
}

llvm::Value* Utils::getValueRelativeToPointer(IRContext* context, llvm::Value* ptr, int offset, llvm::Type* retType) {
    auto* offsetVal = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, offset));
    return Utils::getValueRelativeToPointer(context, ptr, offsetVal, retType);
}

llvm::Value* Utils::getValueRelativeToPointer(IRContext* context, llvm::Value* ptr, llvm::Value* offsetIndex) {
    return Utils::getValueRelativeToPointer(context, ptr, offsetIndex, ptr->getType());
}

llvm::Value* Utils::getValueRelativeToPointer(IRContext* context, llvm::Value* ptr, int offset) {
    auto* offsetVal = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(32, offset));
    return Utils::getValueRelativeToPointer(context, ptr, offsetVal, ptr->getType());
}

/**
 * Inserts a LLVM Value relative to a pointer and a runtime offset in number of elements
 * @param context
 * @param type
 * @param ptr - Base pointer
 * @param offset - Number of elements away (valsize in bits)
 * @param val - Value we are storing
 * @param valSize - Default 64 (8 Bytes)
 */
void Utils::insertRelativeToPointer(IRContext* context, llvm::Type* type, llvm::Value* ptr, llvm::Value* offsetIndex,
                                    llvm::Value* val) {
    auto* offsetPtr = context->Builder->CreateInBoundsGEP(ptr, offsetIndex, "");
    // auto bitcastVal = context->Builder->CreateBitCast(val, offsetPtr->getType());
    context->Builder->CreateStore(val, offsetPtr);
}

void Utils::insertRelativeToPointer(IRContext* context, llvm::Type* type, llvm::Value* ptr, int offset,
                                    llvm::Value* val) {
    auto* offsetPtr = context->Builder->CreateInBoundsGEP(
        ptr, llvm::ConstantInt::get(llvm::Type::getInt32Ty(context->module->getContext()), offset), "");
    insertRelativeToPointer(context, type, ptr, offsetPtr, val);
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
void Utils::insertRelativeToPointer(IRContext* context, llvm::Value* ptr, int offset, llvm::Value* val) {
    int headerBitSize = 32;
    auto offsetIndex = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(headerBitSize, offset, true));
    insertRelativeToPointer(context, val->getType(), ptr, offsetIndex, val);
}

llvm::Type* Utils::convertCuMatTypeToLLVM(IRContext* context, Typing::PRIMITIVE typePrim) {
    llvm::Type* ty;
    switch (typePrim) {
        case Typing::PRIMITIVE::INT: {
            ty = static_cast<llvm::Type*>(llvm::Type::getInt64Ty(context->module->getContext()));
            break;
        }
        case Typing::PRIMITIVE::FLOAT: {
            ty = llvm::Type::getFloatTy(context->module->getContext());
            break;
        }
        case Typing::PRIMITIVE::BOOL: {
            ty = static_cast<llvm::Type*>(llvm::Type::getInt1Ty(context->module->getContext()));
            break;
        }
        default: {
            std::cerr << "Cannot find a valid type" << std::endl;
            // Assign the type to be an integer
            ty = static_cast<llvm::Type*>(llvm::Type::getInt64Ty(context->module->getContext()));
            break;
        }
        case Typing::PRIMITIVE::STRING:
            break;
        case Typing::PRIMITIVE::NONE:
            break;
    }
    return ty;
}

llvm::Value* Utils::getValueFromLLVM(IRContext* context, int val, Typing::PRIMITIVE typePrim, bool isSigned) {
    llvm::Type* type = convertCuMatTypeToLLVM(context, typePrim);
    if (typePrim != Typing::PRIMITIVE::FLOAT) {
        return llvm::ConstantInt::get(context->module->getContext(),
                                      llvm::APInt(type->getPrimitiveSizeInBits(), val, isSigned));
    }
    return nullptr;
}

llvm::Value* Utils::getValueFromLLVM(IRContext* context, float val, Typing::PRIMITIVE typePrim, bool isSigned) {
    llvm::Type* type = convertCuMatTypeToLLVM(context, typePrim);
    if (typePrim == Typing::PRIMITIVE::FLOAT) {
        return llvm::ConstantFP::get(context->module->getContext(), llvm::APFloat(val));
    }
    return nullptr;
}

/**
 * Inserts a LLVM Value relative to a pointer and an offset in number of elements
 * @param context
 * @param ptr - Base pointer
 * @param offset - Number of elements away (valsize in bits)
 * @param val - Value we are storing
 */
void Utils::insertValueAtPointerOffset(Utils::IRContext* context, llvm::Value* ptr, int offset, llvm::Value* val) {
    auto zeroOffset = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context->module->getContext()), 0);
    auto xOffset = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context->module->getContext()), offset);
    auto offsetPtr = context->Builder->CreateInBoundsGEP(ptr, {zeroOffset, xOffset});
    context->Builder->CreateStore(val, offsetPtr);
}

void Utils::insertValueAtPointerOffsetValue(Utils::IRContext* context, llvm::Value* ptr, llvm::Value* offsetValue,
                                            llvm::Value* val) {
    auto zeroOffset = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context->module->getContext()), 0);
    auto offsetPtr = context->Builder->CreateInBoundsGEP(ptr, {zeroOffset, offsetValue});
    context->Builder->CreateStore(val, offsetPtr);
}

/**
 * Get a value from a pointer at an offset
 * @param context
 * @param ptr - Base pointer
 * @param offset - Number of elements away
 * @param name - Name to be stored as in IR
 */
llvm::Value* Utils::getValueFromPointerOffset(Utils::IRContext* context, llvm::Value* ptr, int offset,
                                              std::string name) {
    auto zeroOffset = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context->module->getContext()), 0);
    auto xOffset = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context->module->getContext()), offset);
    auto offsetPtr = context->Builder->CreateInBoundsGEP(ptr, {zeroOffset, xOffset});
    return context->Builder->CreateLoad(offsetPtr, name);
}
