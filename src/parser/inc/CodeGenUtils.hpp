#pragma once

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Type.hpp"

namespace Utils {
static std::map<std::string, llvm::AllocaInst*> AllocSymbolTable;

struct IRContext {
    llvm::Module* module;
    llvm::IRBuilder<>* Builder;
};

struct LLVMMatrixRecord {
    llvm::Value* dataPtr;
    llvm::Value* rank; // Signed
    llvm::Value* numBytes; // Signed
};

void insertRelativeToPointer(IRContext* context, llvm::Type* type, llvm::Value* ptr, int offset, llvm::Value* val);
llvm::Value* getValueRelativeToPointer(IRContext* context, llvm::Type* type, llvm::Value* ptr, int offset);

llvm::AllocaInst* createMatrix(IRContext* context, const Typing::Type& type);
std::unique_ptr<Utils::LLVMMatrixRecord> getMatrixFromPointer(IRContext* context, llvm::AllocaInst* basePtr);
}  // namespace Utils

