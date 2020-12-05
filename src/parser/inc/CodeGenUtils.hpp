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
    llvm::Function* function;
};

struct LLVMMatrixRecord {
    llvm::Value* dataPtr;
    llvm::Value* rank;      // Signed
    llvm::Value* numBytes;  // Signed
};

llvm::Type* convertCuMatTypeToLLVM(IRContext* context, Typing::PRIMITIVE typePrim);

llvm::Value* getValueFromLLVM(IRContext* context, int val, Typing::PRIMITIVE typePrim, bool isSigned);
llvm::Value* getValueFromLLVM(IRContext* context, float val, Typing::PRIMITIVE typePrim, bool isSigned);

llvm::AllocaInst* createMatrix(IRContext* context, const Typing::Type& type);
LLVMMatrixRecord getMatrixFromPointer(IRContext* context, llvm::Value* basePtr);

llvm::Value* getValueRelativeToPointer(IRContext* context, llvm::Value* ptr, int offset);
llvm::Value* getValueRelativeToPointer(IRContext* context, llvm::Value* ptr, int offset, llvm::Type* retType);
llvm::Value* getValueRelativeToPointer(IRContext* context, llvm::Value* ptr, llvm::Value* offsetIndex);
llvm::Value* getValueRelativeToPointer(IRContext* context, llvm::Value* ptr, llvm::Value* offsetIndex,
                                       llvm::Type* retType);

void insertRelativeToPointer(IRContext* context, llvm::Type* type, llvm::Value* ptr, llvm::Value* offsetIndex,
                             llvm::Value* val);
void insertRelativeToPointer(IRContext* context, llvm::Value* ptr, int offset, llvm::Value* val);

llvm::Value* getLength(IRContext* context, llvm::Value* basePtr, const Typing::MatrixType& type);
}  // namespace Utils
