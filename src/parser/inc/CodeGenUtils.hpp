#pragma once

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>

#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "CompilerOptions.hpp"
#include "SymbolTable.hpp"
#include "Type.hpp"
#include "TypeCheckingSymbolTable.hpp"

namespace Utils {
struct IRContext {
    llvm::Module* module;
    llvm::IRBuilder<>* Builder;
    llvm::Function* function;
    SymbolTable* symbolTable;
    CompilerOptions* compilerOptions;
    TypeCheckUtils::TypeCheckingSymbolTable* semanticSymbolTable;
};

struct LLVMMatrixRecord {
    llvm::Value* dataPtr;
    llvm::Value* rank;      // Signed
    llvm::Value* numBytes;  // Signed
    llvm::Value* dimensionPtr;
};

enum FunctionCUDAType { Host, Device };

llvm::Type* convertCuMatTypeToLLVM(IRContext* context, Typing::PRIMITIVE typePrim);

void setNVPTXFunctionType(Utils::IRContext* context, const std::string& funcName, FunctionCUDAType cudeType,
                          llvm::Function* func);

llvm::Value* getValueFromLLVM(IRContext* context, int val, Typing::PRIMITIVE typePrim, bool isSigned);

llvm::Value* getValueFromLLVM(IRContext* context, float val, Typing::PRIMITIVE typePrim, bool isSigned);

llvm::AllocaInst* CreateEntryBlockAlloca(llvm::IRBuilder<>& Builder, const std::string& VarName, llvm::Type* Type);

llvm::Instruction* createMatrix(IRContext* context, const Typing::Type& type);

LLVMMatrixRecord getMatrixFromPointer(IRContext* context, llvm::Value* basePtr);

void insertValueAtPointerOffset(IRContext* context, llvm::Value* ptr, int offset, llvm::Value* val, bool i64);

void insertValueAtPointerOffsetValue(IRContext* context, llvm::Value* ptr, llvm::Value* offsetValue, llvm::Value* val,
                                     bool i1);

llvm::Value* getValueFromPointerOffsetBool(Utils::IRContext* context, llvm::Value* ptr, int offset, const std::string& name);

llvm::Value* getValueFromPointerOffset(IRContext* context, llvm::Value* ptr, int offset, const std::string& name);

llvm::Value* getValueFromPointerOffsetValue(IRContext* context, llvm::Value* ptr, llvm::Value* offsetValue,
                                            const std::string& name);

llvm::Value* getValueFromPointerOffsetValueBool(IRContext* context, llvm::Value* ptr, llvm::Value* offsetValue, const std::string& name);

llvm::Value* getPointerAddressFromOffset(IRContext* context, llvm::Value* ptr, llvm::Value* offset);

llvm::Value* getValueFromIndex(IRContext* context, llvm::Value* ptr, std::shared_ptr<Typing::MatrixType> mat,
                               const std::vector<llvm::Value*>& indicies);
llvm::Value* getValueFromMatrixPtr(IRContext* context, llvm::Value* mPtr, llvm::Value* offset, const std::string& name);

llvm::Value* getValueFromMatrixPtrBool(IRContext* context, llvm::Value* mPtr, llvm::Value* offset, const std::string& name);


void setValueFromMatrixPtr(IRContext* context, llvm::Value* mPtr, llvm::Value* offset, llvm::Value* val);
void setValueFromMatrixPtrBool(IRContext* context, llvm::Value* mPtr, llvm::Value* offset, llvm::Value* val);

llvm::Value* getLength(IRContext* context, llvm::Value* basePtr, const Typing::MatrixType& type);

int getRealIndexOffset(const std::vector<uint>& dimensions, const std::vector<int>& index);

llvm::Value* upcastLiteralToMatrix(Utils::IRContext* context, const Typing::Type  &type, llvm::Value* literalVal);

}  // namespace Utils
