#include "VariableNode.hpp"

#include <numeric>
#include <valarray>
#include "CodeGenUtils.hpp"

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
    this->type = context->symbolTable->getValue(this->name,context->symbolTable->getCurrentFunction())->type;

    if (this->variableSlicing) {
        this->variableSlicing->semanticPass(context);
    } else {
        throw std::runtime_error("[Internal Error] Variable slicing not generated correctly!");
    }
}

llvm::Value* AST::VariableNode::handleSlicing(Utils::IRContext* context, llvm::Value* val) {
    // Create the vectors we will need for storing the intermediaries
    std::vector<std::pair<int, int>> currentIndicies;
    std::vector<std::pair<int, int>> prevIndicies;
    std::vector<std::variant<bool, std::vector<int>>> slicesVec = variableSlicing->slices;

    // Get the type of the matrix we have to be sliced
    Typing::MatrixType* matType = std::get_if<Typing::MatrixType>(&*this->type);
    if (!matType) {
        throw std::runtime_error("[Internal Error] Attempt to slice non-matrix type propagated to codegen");
    }

    // We currently have an awful way of handling slices, we need a vector of pairs that is guaranteed to be
    // filled as we expect
    std::vector<std::pair<int, int>> slices;
    // Give me a full vector of pairs :)
    for (int i = 0; i < matType->dimensions.size(); i++) {
        // If we have a star as the option, we push the entire slice as a (0,dim) pair
        if (slicesVec.size() <= i || std::get_if<bool>(&slicesVec.at(i))) {
            slices.emplace_back(0, matType->dimensions.at(i));
        } else {
            // We should get the element to be sliced and we can check if we have 1 or 2 indicies filled in
            // If one, this means from the index to the dim end
            // If two, nice and simple, just insert
            auto sliceElement = *std::get_if<std::vector<int>>(&slicesVec.at(i));
            if (sliceElement.size() == 2) {
                slices.emplace_back(sliceElement[0], sliceElement[1]);
            } else {
                slices.emplace_back(sliceElement.at(0), matType->dimensions.at(i));
            }
        }
    }

    // Lloyd's algorithm to calculate indicies from slices
    std::vector<int> firstSlices;
    std::vector<uint> slicedMatrixDimensions;

    // For all of the slices, get the first slice elements
    std::transform(slices.begin(), slices.end(), std::back_inserter(firstSlices),
                   [](std::pair<int, int> slice) -> int { return slice.first; });
    std::vector<std::vector<int>> groupIndicies;
    // Place this in as our starting point
    groupIndicies.emplace_back(firstSlices);
    int groupLength = slices[0].second - slices[0].first + 1;
    for (int i = 0; i < slices.size(); i++) {
        // Update the size of the slice dimensions
        slicedMatrixDimensions.emplace_back(abs(slices[i].second - slices[i].first + 1));
        if (i == 0) continue;
        // Get the number of elements in this slice (plus one)
        int number = abs(slices[i].second - slices[i].first) + 1;
        std::vector<std::vector<int>> lastGroupIndicies;
        std::copy(groupIndicies.begin(), groupIndicies.end(), std::back_inserter(lastGroupIndicies));
        groupIndicies = {};
        // Try and add the remaining slices that we have at this level
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

    // We need to convert all of the values into offsets in linear memory (from a vector of size n) i.e. {1,5,2} -> 32
    std::vector<int> offsets;
    offsets.reserve(groupIndicies.size());
    for (const std::vector<int>& index : groupIndicies) {
        offsets.emplace_back(Utils::getRealIndexOffset(matType->dimensions, index));
    }

    // Create a (llvm) vector of size n which contains the offsets. This is the above vector just in llvm memory
    // This could be simplified into a single loop but I don't feel like it and its no real computational cost
    int offsetNumElements =
        std::accumulate(slicedMatrixDimensions.begin(), slicedMatrixDimensions.end(), 1, std::multiplies<int>());
    auto* offsetElementTy = llvm::Type::getInt32Ty(context->module->getContext());
    // Specify the vector type and how many elements we expect it to contain
    llvm::Type* offsetVecTy = llvm::VectorType::get(offsetElementTy, offsetNumElements);
    llvm::Value* emptyVec = llvm::UndefValue::get(offsetVecTy);
    for (int i = 0; i < offsets.size(); i++) {
        // Create offsets and insert to the vector
        llvm::Constant* offsetIndex = llvm::Constant::getIntegerValue(offsetElementTy, llvm::APInt(32, i));
        llvm::Constant* offsetVal = llvm::Constant::getIntegerValue(offsetElementTy, llvm::APInt(32, offsets[i]));
        context->Builder->CreateInsertElement(emptyVec, offsetVal, offsetIndex);
    }

    // Generation of a CuMat Matrix Type
    std::shared_ptr<Typing::MatrixType> slicedMatrixType = std::make_shared<Typing::MatrixType>();
    slicedMatrixType->dimensions = std::move(slicedMatrixDimensions);
    slicedMatrixType->rank = slicedMatrixType->dimensions.size();
    slicedMatrixType->primType = matType->primType;

    // Create a matrix for the sliced element
    auto* slicedMatrix = Utils::createMatrix(context, *slicedMatrixType);

    // Get out llvm record of the new matrix
    auto slicedMatrixRecord = Utils::getMatrixFromPointer(context, slicedMatrix);
    auto storedRecord = Utils::getMatrixFromPointer(context, val);

    // Create a LLVM loop to handle the copying from source matrix to destination matrix
    // This should loop over all of the elements in the vector, so we need to keep track of the position in the vector
    auto* Builder = context->Builder;
    llvm::Function* parent = Builder->GetInsertBlock()->getParent();
    llvm::BasicBlock* copyBB = llvm::BasicBlock::Create(Builder->getContext(), "slice.loop", parent);
    llvm::BasicBlock* endBB = llvm::BasicBlock::Create(Builder->getContext(), "slice.done");

    // Iterator for the loop (SSA)
    auto* iterator = new llvm::AllocaInst(offsetElementTy, 0, "", context->Builder->GetInsertBlock());

    llvm::ConstantInt* offsetSizeLLVM =
        llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(32, offsetNumElements, true));
    // Gives us the size of the block we are wishing to copy each time
    auto* blockSize = llvm::ConstantInt::get(context->module->getContext(),
                                             llvm::APInt(32, abs(slices[0].second - slices[0].first + 1)));
    Builder->CreateBr(copyBB);

    Builder->SetInsertPoint(copyBB);
    {
        auto* index = Builder->CreateLoad(iterator);

        // Get the index out of the vector
        auto* realCurrentIndex = context->Builder->CreateExtractElement(emptyVec, index);
        // Convert to a llvm pointer
        auto* srcPtr = Utils::getPointerAddressFromOffset(context, storedRecord.dataPtr, realCurrentIndex);
        // We have the real index for the original matrix, we need to destination -> Just the iteration * blockSize
        auto* destIndex = Builder->CreateMul(index, blockSize);
        auto* destPtr = Utils::getPointerAddressFromOffset(context, slicedMatrixRecord.dataPtr, destIndex);

        // Update counter
        auto* next = Builder->CreateAdd(
            index, llvm::ConstantInt::get(context->module->getContext(), llvm::APInt{32, 1, true}), "sliceItt");
        Builder->CreateStore(next, iterator);

        // Copy the data from the real to the new - Do this after the increment so we don't need to recompute
        context->Builder->CreateMemCpy(
            destPtr, slicedMatrixRecord.dataPtr->getPointerAlignment(context->module->getDataLayout()), srcPtr,
            storedRecord.dataPtr->getPointerAlignment(context->module->getDataLayout()), blockSize);

        // Test if completed list, if not, add another element!
        auto* done = Builder->CreateICmpUGE(next, offsetSizeLLVM);
        Builder->CreateCondBr(done, endBB, copyBB);
    }
    // Fill in the entry points and stuff
    parent->getBasicBlockList().push_back(endBB);
    Builder->SetInsertPoint(endBB);

    // Return the sliced matrix value
    return slicedMatrix;
}