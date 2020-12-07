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
    llvm::Type* ty = matType.getLLVMType(context);

    auto* headerTy = llvm::IntegerType::getInt64Ty(context->module->getContext());

    llvm::ArrayType* matDataType = llvm::ArrayType::get(ty, matLength);

    llvm::ConstantInt* matSizeLLVM =
        llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, matLength, false));
    llvm::ConstantInt* matSizeHeader =
        llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, matType.rank + 3, false));

    // Allocation pointers for header and end
    auto matAlloc = context->Builder->CreateAlloca(matDataType, 0, matSizeLLVM, "matVar");

    std::vector<llvm::Type*> headerTypes;
    headerTypes.push_back(matDataType);
    headerTypes.push_back(headerTy);        // Rank
    headerTypes.push_back(headerTy);     // # of bytes
    for(int i = 0; i < matType.rank; i++){  // Dimensions
        headerTypes.push_back(headerTy);
    }

    auto* matHeaderType = llvm::StructType::create(headerTypes);

    auto matHeaderAlloc = context->Builder->CreateAlloca(matHeaderType, 0, matSizeHeader, "headerAlloc");

    auto rank = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, matType.rank));
    auto numBytes = llvm::ConstantInt::get(context->module->getContext(),
                                           llvm::APInt(64, (matType.getLength() * matType.offset()) / 8));

    // Insert header into data
    insertRelativeToPointer(context, matHeaderAlloc, 0, matAlloc);  // For the pointer to the data
    insertRelativeToPointer(context, matHeaderAlloc, 1, rank);      // For rank of matrix
    insertRelativeToPointer(context, matHeaderAlloc, 2, numBytes);  // For number of bytes in matrix

    // For the matrix dimensionality
    for (int i = 0; i < matType.rank; i++) {
        auto val = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, matType.dimensions.at(i)));

        // Offset of 3 from before
        insertRelativeToPointer(context, matHeaderAlloc, i + 3, val);
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
    auto ptrOffset = llvm::GetElementPtrInst::Create(type->getPointerElementType(), ptr, {zero, offsetIndex}, "",
                                                     context->Builder->GetInsertBlock());
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
    int headerBitSize = 64;
    auto* offsetPtr = context->Builder->CreateInBoundsGEP(ptr, offsetIndex, "");
    auto bitcastVal = context->Builder->CreateBitCast(val, offsetPtr->getType());
    context->Builder->CreateStore(bitcastVal, offsetPtr);
}

void Utils::insertRelativeToPointer(IRContext* context, llvm::Type* type, llvm::Value* ptr, int offset, llvm::Value* val) {
    int headerBitSize = 64;
    auto* offsetPtr = context->Builder->CreateInBoundsGEP(ptr, llvm::ConstantInt::get(llvm::Type::getInt32Ty(context->module->getContext()), offset), "");
    auto bitcastVal = context->Builder->CreateBitCast(val, offsetPtr->getType());
    context->Builder->CreateStore(bitcastVal, offsetPtr);
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
    int headerBitSize = 64;
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
