#pragma once

#include <llvm-10/llvm/ADT/StringRef.h>
#include <llvm-10/llvm/IR/LLVMContext.h>
#include <llvm-10/llvm/IR/Module.h>

#include <memory>

struct LLVMUtils {
    static llvm::LLVMContext TheContext;
    static std::unique_ptr<llvm::Module> module;
};