#pragma once

#include <llvm/IR/Instructions.h>
#include <llvm/IR/IRBuilder.h>

#include <map>
#include <string>
#include <vector>

namespace Utils {
static std::map<std::string, llvm::AllocaInst*> AllocSymbolTable;

llvm::AllocaInst* generateMatrixAllocation(llvm::Type * ty,
                                           const std::vector<int>& dimensions,
                                           llvm::IRBuilder<> * Builder);
}  // namespace Utils