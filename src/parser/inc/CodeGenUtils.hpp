#pragma once

#include <llvm/IR/Instructions.h>

#include <map>
#include <string>

namespace Utils {
    static std::map<std::string, llvm::AllocaInst*> AllocSymbolTable;
}  // namespace Utils