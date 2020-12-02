#pragma once

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>

#include <Type.hpp>

#include <map>
#include <string>
#include <vector>

namespace Utils {
static std::map<std::string, llvm::AllocaInst*> AllocSymbolTable;

struct IRContext {
    llvm::Module* module;
    llvm::IRBuilder<>* Builder;
};

llvm::AllocaInst* createMatrix(IRContext* context, const Typing::Type &type);

}  // namespace Utils