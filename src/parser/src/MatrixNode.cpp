#include "MatrixNode.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Type.h>

#include <iostream>
#include <numeric>

#include "CodeGenUtils.hpp"
#include "DimensionPass.hpp"
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

    int trueIndex = 0;  // True index measures the current index plus any of the j's
    // Compress literal nodes to matrix representation
    if (auto primType = std::get_if<Typing::MatrixType>(&*this->type)) {
        llvm::Type* elType = primType->getLLVMPrimitiveType(context);
        llvm::ArrayType* arrayType = llvm::ArrayType::get(elType, this->data.size());
        std::vector<llvm::Constant*> values;

        for (int i = 0; i < this->data.size(); i++) {
            auto element = this->data.at(i);

            // Check if we can naively do this without causing recursive check -> Yes its not technically a matrix
            // Yes this violates our idea of every type being a matrix but its equivalent fuck it
            auto* literal = std::get_if<Typing::MatrixType>(&*element->type);
            if (literal->rank == 0) {
                // Let's just generate code for the literal itself -> This returns a single value
                auto* elementLLVMMat = static_cast<llvm::Constant*>(element->codeGen(context));
                values.push_back(elementLLVMMat);
                //                Utils::insertValueAtPointerOffset(context, matRecord.dataPtr, i, elementLLVMMat,
                //                false);
                trueIndex++;
            } else {
                // We have encountered a non-literal value -> This when evaluated WILL return a matrix
                // Because it returns a matrix, we want to copy all of its data into our new matrix
                llvm::Value* matLLVMVal = element->codeGen(context);
                auto matElementRecord = Utils::getMatrixFromPointer(context, matLLVMVal);
                // All other types are 64-bit
                int numElements = literal->getLength();
                for (int j = 0; j < numElements; j++) {
                    auto* oldLLVMVal =
                        Utils::getValueFromPointerOffset(context, matElementRecord.dataPtr, j, "matCpyOut");
                    llvm::Type* tyInt = llvm::Type::getInt64Ty(context->module->getContext());
                    auto* trueIndexLLVM = llvm::ConstantInt::get(tyInt, trueIndex + j);
                    Utils::setValueFromMatrixPtr(context, matRecord.dataPtr, trueIndexLLVM, oldLLVMVal);
                }
                trueIndex += numElements;
            }
        }
        auto* i32Ty = llvm::Type::getInt32Ty(context->module->getContext());
        llvm::Constant* init = llvm::ConstantArray::get(arrayType, values);

        auto * dataAllocSize = llvm::ConstantExpr::getTruncOrBitCast(llvm::ConstantExpr::getSizeOf(arrayType), i32Ty);
        auto* arr = llvm::CallInst::CreateMalloc(context->Builder->GetInsertBlock(), i32Ty, arrayType,
                                                 dataAllocSize, nullptr, nullptr, "");
        context->Builder->Insert(arr, "matTmpData");
        context->Builder->CreateStore(init, arr);
        context->Builder->CreateMemCpy(
            matRecord.dataPtr, matRecord.dataPtr->getPointerAlignment(context->module->getDataLayout()), arr,
            arr->getPointerAlignment(context->module->getDataLayout()), llvm::ConstantExpr::getSizeOf(arrayType));
        auto * freeMem = llvm::CallInst::CreateFree(arr, context->Builder->GetInsertBlock());
        context->Builder->Insert(freeMem);
        //        llvm::BasicBlock* addBB =
        //            llvm::BasicBlock::Create(context->Builder->getContext(), "matInitEntry", context->function);
        //        llvm::BasicBlock* endBB = llvm::BasicBlock::Create(context->Builder->getContext(), "matInitEnd");
        //
        //        auto indexAlloca = Utils::CreateEntryBlockAlloca(*context->Builder, "startIndex",
        //                                                         llvm::Type::getInt32Ty(context->Builder->getContext()));
        //        llvm::Constant* matIndexEnd = llvm::ConstantInt::get(i32Ty, values.size());
        //        // parent->getBasicBlockList().push_back(addBB);
        //        context->Builder->CreateBr(addBB);
        //
        //        context->Builder->SetInsertPoint(addBB);
        //        {
        //            auto* index = context->Builder->CreateLoad(indexAlloca, "index");
        //
        //            llvm::Value* arrElement = Utils::getValueFromPointerOffsetValue(context, arr, index,
        //            "getArrConst"); Utils::insertValueAtPointerOffsetValue(context, matRecord.dataPtr, index,
        //            arrElement, false);
        //
        //            // Update counter
        //            auto* next = context->Builder->CreateAdd(
        //                index, llvm::ConstantInt::get(context->module->getContext(), llvm::APInt{32, 1, true}),
        //                "inc");
        //            context->Builder->CreateStore(next, indexAlloca);
        //
        //            // Test if completed list
        //            auto* done = context->Builder->CreateICmpSGE(next, matIndexEnd);
        //            context->Builder->CreateCondBr(done, endBB, addBB);
        //        }
        //
        //        context->function->getBasicBlockList().push_back(endBB);
        //        context->Builder->SetInsertPoint(endBB);
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

    std::vector<uint> dimensions = this->getDimensions(); // Maybe use dimensions of inner matrix?

    this->type = TypeCheckUtils::makeMatrixType(dimensions, primType);
}

void AST::MatrixNode::dim_subpass(std::vector<uint>& apparent_dim, std::vector<uint>& size,
                                  const std::vector<uint>& nodedims, int sep) {
    if (nodedims.size() > sep) {
        dimension_error("Expression rank larger than separator", this);
    }
    while (size.size() < nodedims.size() || size.size() < sep) {
        size.emplace_back(0);
    }
    for (int i = 0; i < nodedims.size(); i++) {
        size[i] += nodedims[i];
    }
    for (int i = 0; i < sep - 1; i++) {
        if (i + 1 > apparent_dim.size()) {
            apparent_dim.emplace_back(size[i]);
        }
        if (size[i] != apparent_dim[i]) {
            dimension_error(std::string{"Dimension value mismatched. Expected: "} + std::to_string(apparent_dim[i]) +
                                ", got: " + std::to_string(size[i]),
                            this);
        }
        size[i] = 0;
        if (i + 1 >= nodedims.size()) {
            size[i + 1] += 1;
        }
    }
}

void AST::MatrixNode::dimensionPass(Analysis::DimensionSymbolTable* nt) {
    for (auto& elem : this->data) {
        elem->dimensionPass(nt);
    }
    std::vector<uint> apparent_dim{};
    std::vector<uint> size{};

    {
        auto sep = this->separators.begin();
        for (int i = 0; i < this->data.size() - 1; i++) {
            const auto& elem = data[i];
            const auto* type = std::get_if<Typing::MatrixType>(&*elem->type);
            if (type) {
                const auto& dims = type->dimensions;
                dim_subpass(apparent_dim, size, dims, *(sep++));
            }
        }
    }
    {
        // have to do this separately as sep is one element shorter than data
        const auto& elem = data[data.size() - 1];
        const auto* type = std::get_if<Typing::MatrixType>(&*elem->type);
        if (type) {
            const auto& dims = type->dimensions;
            dim_subpass(apparent_dim, size, dims, size.size());
        }
        apparent_dim.emplace_back(size.back());
    }

    auto* type = std::get_if<Typing::MatrixType>(&*this->type);
    if (type) {
        type->dimensions = apparent_dim;
        type->rank = apparent_dim.size();
    }

    std::vector<std::shared_ptr<ExprNode>> new_vector{};

    for (auto& elem : this->data) {
        if (elem->isConst()) {
            for (auto& inner_elem : elem->constData(elem)) {
                new_vector.emplace_back(std::move(inner_elem));
            }
        } else {
            new_vector.emplace_back(std::move(elem));
        }
    }
    this->data = std::move(new_vector);
}

std::string AST::MatrixNode::toTree(const std::string& prefix, const std::string& childPrefix) const {
    return prefix + "Matrix";
}

bool AST::MatrixNode::isConst() const noexcept {
    auto acc = true;
    // I could use std::transform_reduce, but CBA...
    for (auto& a : this->data) {
        acc &= a->isConst();
    }
    return acc;
}

std::vector<std::shared_ptr<AST::ExprNode>> AST::MatrixNode::constData(std::shared_ptr<AST::ExprNode>& me) const {
    if (!this->isConst()) {
        throw std::runtime_error("attempt to access constData on non-const node");
    }
    return this->data;
}