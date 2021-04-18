#include "CodeGenUtils.hpp"

#include <algorithm>
#include <iostream>

// TODO: USE NEW getLLVMType change to get the proper type
llvm::Instruction* Utils::createMatrix(Utils::IRContext* context, const Typing::Type& type) {
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
    llvm::Type* matHeaderType;
    if(!context->symbolTable->headerTypeGenerated){
        matHeaderType = matType.getLLVMType(context);
        context->symbolTable->matHeaderType = matHeaderType;
        context->symbolTable->headerTypeGenerated = true;
    }else{
        matHeaderType = context->symbolTable->matHeaderType;
    }

    // Create a type for the actual data of the matrix + length info
    llvm::ArrayType* matDataType = llvm::ArrayType::get(ty, 0);
    llvm::ConstantInt* matSizeLLVM =
        llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, matLength, false));

    // Rank / dimensionality meta-data
    auto rank = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, matType.rank));
    auto numBytes = llvm::ConstantInt::get(context->module->getContext(),
                                           llvm::APInt(64, (matType.getLength() * matType.offset()) / 8));

    // Constant zero offset used for base ptr
    auto zeroOffset = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context->module->getContext()), 0);

    // Allocation of the matrix data
    auto* intPtrType = llvm::Type::getInt32Ty(context->module->getContext());
    llvm::Constant* matAllocaSize = llvm::ConstantInt::get(intPtrType, matLength);
    // This will by default be i64, need to cast to i32 (I think its safe)
    matAllocaSize = llvm::ConstantExpr::getTruncOrBitCast(matAllocaSize, intPtrType);
    auto* matAlloc = llvm::CallInst::CreateMalloc(context->Builder->GetInsertBlock(), intPtrType, matDataType,
                                                  matAllocaSize, nullptr, nullptr, "bitcast");
    context->Builder->Insert(matAlloc, "matArrData");

    // Pointer for the array itself accessible for loading
    auto matAllocPtr = context->Builder->CreateGEP(matAlloc, zeroOffset, "matArrPtr");

    // We need an integer pointer type for the address
    intPtrType = llvm::Type::getInt64Ty(context->module->getContext());
    llvm::Constant* matHeaderAllocaSize = llvm::ConstantInt::get(intPtrType, 4);
    matHeaderAllocaSize = llvm::ConstantExpr::getTruncOrBitCast(matHeaderAllocaSize, intPtrType);
    auto* matHeaderAlloc = llvm::CallInst::CreateMalloc(context->Builder->GetInsertBlock(), intPtrType, matHeaderType,
                                                        matHeaderAllocaSize, nullptr, nullptr, "bitcast");
    // LLVM requires us to actually insert the instruction when using CallInst
    context->Builder->Insert(matHeaderAlloc, "matStruct");

    insertValueAtPointerOffset(context, matHeaderAlloc, 0, matAllocPtr, false);
    insertValueAtPointerOffset(context, matHeaderAlloc, 1, rank, false);
    insertValueAtPointerOffset(context, matHeaderAlloc, 2, numBytes, false);

    // For the matrix dimensionality
    llvm::ArrayType* matDimensionType = llvm::ArrayType::get(llvm::Type::getInt64Ty(context->module->getContext()), 0);
    // Allocation of the matrix data
    llvm::Constant* matDimAllocaSize = llvm::ConstantExpr::getSizeOf(matDimensionType);
    // This will by default be i64, need to cast to i32 (I think its safe)
    matDimAllocaSize = llvm::ConstantExpr::getTruncOrBitCast(matDimAllocaSize, intPtrType);
    auto* matDimAlloc = llvm::CallInst::CreateMalloc(context->Builder->GetInsertBlock(), intPtrType, matDimensionType,
                                                     matDimAllocaSize, nullptr, nullptr, "bitcast");
    context->Builder->Insert(matDimAlloc, "matArrData");

    for (int i = 0; i < matType.rank; i++) {
        auto val = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, matType.dimensions.at(i)));

        // Offset of 3 from before
        insertValueAtPointerOffset(context, matDimAlloc, i, val, false);
    }

    insertValueAtPointerOffset(context, matHeaderAlloc, 3, matDimAlloc, false);

    return matHeaderAlloc;
}

// We need to generate meta-data for NVPTX
// This is 100% stolen from https://stackoverflow.com/questions/40082378/how-to-generate-metadata-for-llvm-ir
// as it is someone asking how to do this exact problem :)
void Utils::setNVPTXFunctionType(Utils::IRContext* context, const std::string& funcName, FunctionCUDAType cudeType,
                                 llvm::Function* func) {
    // Vector to store the tuple operations
    llvm::SmallVector<llvm::Metadata*, 3> ops;
    // We reference the type first from the global llvm symbol lookup rather than internal
    // as then we can guarantee we haven't messed up thus far!
    llvm::GlobalValue* funcGlob = context->module->getNamedValue(funcName);
    if (!funcGlob) {
        throw std::runtime_error("[Internal Error] Could not find function to generate metadata for!");
    }

    // Push the function reference
    ops.push_back(llvm::ValueAsMetadata::getConstant(funcGlob));
    // Push the type of the function (device or kernel)
    switch (cudeType) {
        case Host: {
            ops.push_back(llvm::MDString::get(context->module->getContext(), "kernel"));
            break;
        }
        case Device: {
            ops.push_back(llvm::MDString::get(context->module->getContext(), "kernel"));
            break;
        }
    }

    // We need an i64Ty to tell nvptx what API to use (I think)
    llvm::Type* i64ty = llvm::Type::getInt64Ty(context->module->getContext());
    llvm::Constant* one = llvm::ConstantInt::get(i64ty, 1);
    ops.push_back(llvm::ValueAsMetadata::getConstant(one));

    // Generate the tuple with operands and attach it to the function as metadata
    auto* node = llvm::MDTuple::get(context->module->getContext(), ops);
    context->symbolTable->getNVVMMetadata()->addOperand(node);
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

    llvm::Value* dataPtr = Utils::getValueFromPointerOffset(context, basePtr, 0, "dataPtr");
    llvm::Value* rank = Utils::getValueFromPointerOffset(context, basePtr, 1, "rankVal");
    llvm::Value* numBytes = Utils::getValueFromPointerOffset(context, basePtr, 2, "numBytesVal");
    llvm::Value* dimensions = Utils::getValueFromPointerOffset(context, basePtr, 3, "dimensionsPtr");
    return {dataPtr, rank, numBytes, dimensions};
}

llvm::Type* Utils::convertCuMatTypeToLLVM(IRContext* context, Typing::PRIMITIVE typePrim) {
    llvm::Type* ty;
    switch (typePrim) {
        case Typing::PRIMITIVE::INT: {
            ty = static_cast<llvm::Type*>(llvm::Type::getInt64Ty(context->module->getContext()));
            break;
        }
        case Typing::PRIMITIVE::FLOAT: {
            ty = llvm::Type::getDoubleTy(context->module->getContext());
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
void Utils::insertValueAtPointerOffset(Utils::IRContext* context, llvm::Value* ptr, int offset, llvm::Value* val,
                                       bool i64) {
    // Handle the case where we have an i64
    if (i64) offset *= 2;

    auto zeroOffset = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context->module->getContext()), 0);
    auto xOffset = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context->module->getContext()), offset);
    auto offsetPtr = context->Builder->CreateInBoundsGEP(ptr, {zeroOffset, xOffset}, "insertPtr");
    context->Builder->CreateStore(val, offsetPtr);
}

void Utils::insertValueAtPointerOffsetValue(Utils::IRContext* context, llvm::Value* ptr, llvm::Value* offsetValue,
                                            llvm::Value* val, bool i64) {
    // Handle the case where we have an i64
    if (i64)
        offsetValue = context->Builder->CreateMul(
            offsetValue, llvm::ConstantInt::get(llvm::Type::getInt32Ty(context->module->getContext()), 2));

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
                                              const std::string& name) {
    auto zeroOffset = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context->module->getContext()), 0);
    auto xOffset = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context->module->getContext()), offset);
    auto offsetPtr = context->Builder->CreateInBoundsGEP(ptr, {zeroOffset, xOffset}, "getPtr");
    return context->Builder->CreateLoad(offsetPtr, name);
}

llvm::Value* Utils::getValueFromPointerOffsetValue(Utils::IRContext* context, llvm::Value* ptr,
                                                   llvm::Value* offsetValue, const std::string& name) {
    auto zeroOffset = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context->module->getContext()), 0);
    auto offsetPtr = context->Builder->CreateInBoundsGEP(ptr, {zeroOffset, offsetValue});
    return context->Builder->CreateLoad(offsetPtr, name);
}

llvm::Value* Utils::getValueFromMatrixPtr(Utils::IRContext* context, llvm::Value* mPtr, llvm::Value* offset,
                                          const std::string& name) {
    auto* dataPtr = getValueFromPointerOffset(context, mPtr, 0, "dataPtr");
    return getValueFromPointerOffsetValue(context, dataPtr, offset, "matValue");
}

/**
 * We hardcode for N = 1,2,3 and then provide general llvm code for N > 3
 * @param context
 * @param ptr
 * @param mat
 * @param indicies
 * @return
 */
llvm::Value* Utils::getValueFromIndex(Utils::IRContext* context, llvm::Value* ptr,
                                      std::shared_ptr<Typing::MatrixType> mat,
                                      const std::vector<llvm::Value*>& indicies) {
    switch (mat->rank) {
        case 2: {
            // offset = n2 + N2*n1
            auto* mult = context->Builder->CreateMul(
                indicies.at(0),
                getValueFromLLVM(context, static_cast<int>(mat->dimensions.at(1)), Typing::PRIMITIVE::INT, false));
            auto* offset = context->Builder->CreateAdd(indicies.at(1), mult);
            return getValueFromMatrixPtr(context, ptr, offset, "");
        }
        case 3: {
            // This is the last value that we hardcode the offset
            // offset = n3 + N3 * (n2 + N2 * n1)
            // offset = n3 + N3 * (n2 + a)
            // offset = n3 + N3 * (b+a)
            auto* a = context->Builder->CreateMul(
                indicies.at(0),
                getValueFromLLVM(context, static_cast<int>(mat->dimensions.at(1)), Typing::PRIMITIVE::INT, false));
            auto* b = context->Builder->CreateAdd(indicies.at(1), a);
            auto* c = context->Builder->CreateMul(
                getValueFromLLVM(context, static_cast<int>(mat->dimensions.at(2)), Typing::PRIMITIVE::INT, false), b);
            auto* offset = context->Builder->CreateAdd(indicies.at(2), c);
            return getValueFromMatrixPtr(context, ptr, offset, "");
        }
        default: {
            // TODO: implement
            throw std::runtime_error("Accessing matrix elements above rank 3 matrices is not currently supported!");
        }
    }
}

void Utils::setValueFromMatrixPtr(Utils::IRContext* context, llvm::Value* mPtr, llvm::Value* offset, llvm::Value* val) {
    auto* dataPtr = getValueFromPointerOffset(context, mPtr, 0, "dataPtr");
    insertValueAtPointerOffsetValue(context, dataPtr, offset, val, false);
}

llvm::AllocaInst* Utils::CreateEntryBlockAlloca(llvm::IRBuilder<>& Builder, const std::string& VarName,
                                                llvm::Type* Type) {
    llvm::IRBuilder<> TmpB(&Builder.GetInsertBlock()->getParent()->getEntryBlock(),
                           Builder.GetInsertBlock()->getParent()->getEntryBlock().begin());
    return TmpB.CreateAlloca(Type, nullptr, VarName);
}

/**
 * Outputs the offset for a given index and matrix of arbitrary dimension
 * https://en.wikipedia.org/wiki/Row-_and_column-major_order
 * @param dimensions
 * @param index
 * @return
 */
int Utils::getRealIndexOffset(const std::vector<uint>& dimensions, const std::vector<int>& index) {
    int offset = 0;
    for (int k = 1; k <= dimensions.size(); k++) {
        int partialSum = 1;
        for (int l = k + 1; l < dimensions.size(); l++) {
            partialSum *= dimensions.at(l);
        }
        partialSum *= index.at(k - 1);
        offset += partialSum;
    }
    return offset;
}

llvm::Value* Utils::getPointerAddressFromOffset(IRContext* context, llvm::Value* ptr, llvm::Value* offset) {
    auto zeroOffset = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context->module->getContext()), 0);
    return context->Builder->CreateInBoundsGEP(ptr, {zeroOffset, offset});
}

llvm::Value* Utils::upcastLiteralToMatrix(Utils::IRContext* context, const Typing::Type  &type, llvm::Value* literalVal){
    auto* retMatrix = Utils::createMatrix(context, type);
    auto retRecord = Utils::getMatrixFromPointer(context, retMatrix);
    Utils::insertValueAtPointerOffset(context, retRecord.dataPtr, 0, literalVal, false);
    return retMatrix;
}