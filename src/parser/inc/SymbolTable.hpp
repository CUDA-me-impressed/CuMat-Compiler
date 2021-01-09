#pragma once

#include <llvm/IR/Value.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Type.hpp"

namespace Utils {

struct SymbolTableEntry {
    std::shared_ptr<Typing::Type> type;
    llvm::Value* llvmVal;
};

class SymbolTable {
   private:
    // Vector that stores names of variables along with the depth we find them
    std::map<std::string, std::map<std::string, SymbolTableEntry>> data;
    std::vector<std::string> functionStack; // Used to identify which function we should be inside

   public:
    std::shared_ptr<SymbolTableEntry> getValue(const std::string& symbolName,
                                               const std::string& funcName,
                                               const std::string& funcNamespace = "");

    void setValue(std::shared_ptr<Typing::Type> type, llvm::Value* storeVal, const std::string& symbolName,
                  const std::string& funcName, const std::string& funcNamespace = "");
    void updateValue(llvm::Value* value, const std::string& symbolName,
                     const std::string& funcName, const std::string& funcNamespace = "");

    bool inSymbolTable(const std::string& symbolName, const std::string& funcName, const std::string& funcNamespace = "");

    void addFunction(const std::string &funcName, const std::string &funcNamespace = "");
    void escapeFunction();
    const std::string& getCurrentFunction();
};
}  // namespace Utils