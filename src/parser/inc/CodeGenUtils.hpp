#pragma once

#include <llvm-10/llvm/IR/Instructions.h>

#include <map>
#include <string>

namespace Utils {
    static std::map<std::string, llvm::AllocaInst*> AllocSymbolTable;
}  // namespace Utils