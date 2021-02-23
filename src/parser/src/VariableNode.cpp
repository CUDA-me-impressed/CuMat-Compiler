#include "VariableNode.hpp"
#include <valarray>

llvm::Value* AST::VariableNode::codeGen(Utils::IRContext* context) {
    llvm::Value* storeVal =
        context->symbolTable->getValue(this->name, context->symbolTable->getCurrentFunction())->llvmVal;
    // If we have a slicing operation we apply it now
    if (this->variableSlicing) {
        storeVal = handleSlicing(context, storeVal);
    }
    return storeVal;
}

void AST::VariableNode::semanticPass(Utils::IRContext* context) {
    if (this->variableSlicing) {
        this->variableSlicing->semanticPass(context);
    } else {
        throw std::runtime_error("[Internal Error] Variable slicing not generated correctly!");
    }
}

llvm::Value* AST::VariableNode::handleSlicing(Utils::IRContext* context, llvm::Value* val) {
    std::vector<std::pair<int, int>> currentIndicies;
    std::vector<std::pair<int, int>> prevIndicies;
    std::vector<std::variant<bool, std::vector<int>>> slicesVec = variableSlicing->slices;

    Typing::MatrixType* matType = std::get_if<Typing::MatrixType>(&*this->type);
    if (!matType) {
        throw std::runtime_error("[Internal Error] Attempt to slice non-matrix type propagated to codegen");
    }

    std::vector<std::pair<int, int>> slices;
    // Give me a full vector of pairs :)
    for (int i = 0; i < matType->dimensions.size(); i++) {
        if (slicesVec.size() <= i || std::get_if<bool>(&slicesVec.at(i))) {
            slices.emplace_back(0, matType->dimensions.at(i));
        } else {
            auto sliceElement = *std::get_if<std::vector<int>>(&slicesVec.at(i));
            if (sliceElement.size() == 2) {
                slices.emplace_back(sliceElement.at(0), sliceElement.emplace_back(1));
            } else {
                slices.emplace_back(sliceElement.at(0), matType->dimensions.at(i));
            }
        }
    }

    // Lloyd's algorithm to calculate indicies from slices
    std::vector<int> firstSlices;
    std::vector<uint> slicedMatrixDimensions;

    std::transform(slices.begin(), slices.end(), std::back_inserter(firstSlices),
                   [](std::pair<int, int> slice) -> int { return slice.first; });
    std::vector<std::vector<int>> groupIndicies;
    groupIndicies.emplace_back(firstSlices);
    int groupLength = slices[0].second - slices[0].first + 1;
    for (int i = 0; i < slices.size(); i++) {
        // Update the size of the slice dimensions
        slicedMatrixDimensions.emplace_back(abs(slices[i].second - slices[i].first + 1));
        if (i == 0) continue;
        int number = abs(slices[i].second - slices[i].first) + 1;
        std::vector<std::vector<int>> lastGroupIndicies;
        std::copy(groupIndicies.begin(), groupIndicies.end(), std::back_inserter(lastGroupIndicies));
        groupIndicies = {};
        for (int x = 0; x < number; x++) {
            for (auto gi : groupIndicies) {
                int j = 0;
                std::transform(gi.begin(), gi.end(), gi.begin(), [&](int c) -> int {
                  j++;
                  return (j == i) ? c + x : c;
                });
            }
        }
    }

    // We need to convert all of the values into definitive offsets
    std::vector<int> offsets;
    offsets.reserve(groupIndicies.size());
    for (const std::vector<int>& index : groupIndicies) {
        offsets.emplace_back(Utils::getRealIndexOffset(matType->dimensions, index));
    }

    // Create a vector of size n which contains the offsets
    int offsetNumElements =
        std::accumulate(slicedMatrixDimensions.begin(), slicedMatrixDimensions.end(), 1, std::multiplies<int>());
    auto* offsetElementTy = llvm::Type::getInt32Ty(context->module->getContext());
    llvm::Type* offsetVecTy = llvm::VectorType::get(offsetElementTy, offsetNumElements);
    llvm::Value* emptyVec = llvm::UndefValue::get(offsetVecTy);
    for (int i = 0; i < offsets.size(); i++) {
        llvm::Constant* offsetIndex = llvm::Constant::getIntegerValue(offsetElementTy, llvm::APInt(32, i));
        llvm::Constant* offsetVal = llvm::Constant::getIntegerValue(offsetElementTy, llvm::APInt(32, offsets[i]));
        context->Builder->CreateInsertElement(emptyVec, offsetVal, offsetIndex);
    }

    // Generation of a CuMat Matrix Type
    std::shared_ptr<Typing::MatrixType> slicedMatrixType = std::make_shared<Typing::MatrixType>();
    slicedMatrixType->dimensions = std::move(slicedMatrixDimensions);
    slicedMatrixType->rank = slicedMatrixType->dimensions.size();
    slicedMatrixType->primType = matType->primType;

    auto* slicedMatrix = Utils::createMatrix(context, *slicedMatrixType);
    // Get out llvm record of the new matrix
    auto slicedMatrixRecord = Utils::getMatrixFromPointer(context, slicedMatrix);

    /*
     * Create a LLVM loop to handle the copying from source matrix to destination matrix
     */
    auto* Builder = context->Builder;
    llvm::Function* parent = Builder->GetInsertBlock()->getParent();
    llvm::BasicBlock* copyBB = llvm::BasicBlock::Create(Builder->getContext(), "slice.loop", parent);
    llvm::BasicBlock* endBB = llvm::BasicBlock::Create(Builder->getContext(), "slice.done");

    auto* iterator = new llvm::AllocaInst(offsetElementTy, 0, "", context->Builder->GetInsertBlock());

    llvm::ConstantInt* offsetSizeLLVM =
        llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, offsetNumElements, false));
    // Gives us the size of the block we are wishing to copy each time
    auto* blockSize = llvm::ConstantInt::get(context->module->getContext(),
                                             llvm::APInt(32, abs(slices[0].second - slices[0].first + 1)));
    Builder->CreateBr(copyBB);

    Builder->SetInsertPoint(copyBB);
    {
        auto* index = Builder->CreateLoad(iterator);

        // Get the index out of the vector
        auto* realCurrentIndex = context->Builder->CreateExtractElement(emptyVec, index);
        // Update counter
        auto* next = Builder->CreateAdd(
            index, llvm::ConstantInt::get(context->module->getContext(), llvm::APInt{64, 1, true}), "sliceItt");
        Builder->CreateStore(next, iterator);

        // Copy the data from the real to the new - Do this after the increment so we don't need to recompute

        // Test if completed list
        auto* done = Builder->CreateICmpUGE(next, offsetSizeLLVM);
        Builder->CreateCondBr(done, endBB, copyBB);
    }

    parent->getBasicBlockList().push_back(endBB);
    Builder->SetInsertPoint(endBB);

    //    llvm::ArrayType* offsetTy =
    //        llvm::ArrayType::get(llvm::Type::getInt64Ty(context->module->getContext()), offsetNumElements);

    //    auto* offsetArr = context->Builder->CreateAlloca(offsetTy, offsetSizeLLVM, "offsetArr");

    // Store the value of the offsets in the array

    return slicedMatrix;
}