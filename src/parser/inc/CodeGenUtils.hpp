#pragma once

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>

#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Type.hpp"

namespace Utils {
static std::map<std::string, llvm::AllocaInst*> AllocSymbolTable;
static std::map<std::string, std::map<std::vector<std::shared_ptr<Typing::Type>>, llvm::Function*>> funcTable;
struct IRContext {
    llvm::Module* module;
    llvm::IRBuilder<>* Builder;
};

struct LLVMMatrixRecord {
    llvm::Value* dataPtr;
    llvm::Value* rank;      // Signed
    llvm::Value* numBytes;  // Signed
};

void insertRelativeToPointer(IRContext* context, llvm::Type* type, llvm::Value* ptr, int offset, llvm::Value* val);

llvm::Type* convertCuMatTypeToLLVM(IRContext* context, Typing::PRIMITIVE typePrim);

template <typename T>
llvm::Value* getValueFromLLVM(IRContext* context, T val, Typing::PRIMITIVE typePrim, bool isSigned);

llvm::Value* getValueRelativeToPointer(IRContext* context, llvm::Type* type, llvm::Value* ptr, int offset);

llvm::AllocaInst* createMatrix(IRContext* context, const Typing::Type& type);
LLVMMatrixRecord getMatrixFromPointer(IRContext* context, llvm::Value* basePtr);
llvm::Value* getValueRelativeToPointer(IRContext* context, llvm::Type* type, llvm::Value* ptr,
                                       llvm::Value* offsetIndex);
void insertRelativeToPointer(IRContext* context, llvm::Type* type, llvm::Value* ptr, llvm::Value* offsetIndex,
                             llvm::Value* val);
llvm::Value* getLength(IRContext* context, llvm::Value* basePtr, const Typing::MatrixType& type);
}  // namespace Utils
