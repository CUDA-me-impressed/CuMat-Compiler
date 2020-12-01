#pragma once

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>

#include <Type.hpp>

#include <map>
#include <string>
#include <vector>

namespace Utils {
static std::map<std::string, llvm::AllocaInst*> AllocSymbolTable;

class MatrixInterface {
   public:
    llvm::AllocaInst* createMatrix(Typing::Type type, llvm::IRBuilder<>* Builder);
};
}  // namespace Utils