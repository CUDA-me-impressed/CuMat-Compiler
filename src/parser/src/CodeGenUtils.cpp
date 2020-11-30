#pragma once

#include "CodeGenUtils.hpp"

#include <algorithm>

llvm::AllocaInst* Utils::generateMatrixAllocation(
    llvm::Type* ty, const std::vector<int>& dimensions,
    llvm::IRBuilder<>* Builder) {
    int numElements = 1;
    std::for_each(dimensions.begin(), dimensions.end(),
                  [&](int el) { numElements *= el; });
    // Create a store instance for the correct precision and data type
    // Address space set to zero
    return new llvm::AllocaInst(ty, 0, "matVar", Builder->GetInsertBlock());
}