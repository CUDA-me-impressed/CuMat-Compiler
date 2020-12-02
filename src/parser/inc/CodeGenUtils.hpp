#pragma once

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>

#include <map>
#include <string>
#include <vector>
#include <utility>

namespace Utils {
static std::map<std::string, llvm::AllocaInst*> AllocSymbolTable;

struct IRContext {
    llvm::Module* module;
    llvm::IRBuilder<>* Builder;
};

void insertRelativeToPointer(IRContext* context, llvm::Type* type,
                             llvm::Value* ptr, int offset, llvm::Value* val);

llvm::AllocaInst* createMatrix(IRContext* context, const Typing::Type &type);
std::pair<llvm::Value, llvm::Value>
}  // namespace Utils