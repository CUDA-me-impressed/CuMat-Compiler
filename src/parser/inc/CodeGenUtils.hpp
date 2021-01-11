#pragma once

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>

#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "SymbolTable.hpp"
#include "Type.hpp"

namespace Utils {
struct IRContext {
    llvm::Module* module;
    llvm::IRBuilder<>* Builder;
    llvm::Function* function;
    SymbolTable* symbolTable;
};

struct LLVMMatrixRecord {
    llvm::Value* dataPtr;
    llvm::Value* rank;      // Signed
    llvm::Value* numBytes;  // Signed
};

llvm::Type* convertCuMatTypeToLLVM(IRContext* context, Typing::PRIMITIVE typePrim);

llvm::Value* getValueFromLLVM(IRContext* context, int val, Typing::PRIMITIVE typePrim, bool isSigned);
llvm::Value* getValueFromLLVM(IRContext* context, float val, Typing::PRIMITIVE typePrim, bool isSigned);

llvm::AllocaInst* CreateEntryBlockAlloca(llvm::IRBuilder<>& Builder, const std::string& VarName, llvm::Type* Type);

llvm::Instruction* createMatrix(IRContext* context, const Typing::Type& type);
LLVMMatrixRecord getMatrixFromPointer(IRContext* context, llvm::Value* basePtr);

void insertValueAtPointerOffset(IRContext* context, llvm::Value* ptr, int offset, llvm::Value* val);
void insertValueAtPointerOffsetValue(IRContext* context, llvm::Value* ptr, llvm::Value* offsetValue, llvm::Value* val);

llvm::Value* getValueFromPointerOffset(IRContext* context, llvm::Value* ptr, int offset, std::string name);
llvm::Value* getValueFromPointerOffsetValue(IRContext* context, llvm::Value* ptr, llvm::Value* offsetValue,
                                            std::string name);

llvm::Value* getValueFromMatrixPtr(IRContext* context, llvm::Value* mPtr, llvm::Value* offset, std::string name);
void setValueFromMatrixPtr(IRContext* context, llvm::Value* mPtr, llvm::Value* offset, llvm::Value* val);

llvm::Value* getLength(IRContext* context, llvm::Value* basePtr, const Typing::MatrixType& type);
}  // namespace Utils
