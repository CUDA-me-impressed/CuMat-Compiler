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

struct FunctionTableEntry {
    llvm::Function* func;
};

class FunctionParamCompare {
   public:
    bool operator() (const std::shared_ptr<Typing::Type>& l, const std::shared_ptr<Typing::Type>& r) {
        // This is the worst possible way to do this but I really really really am out of options here fuck it
        Typing::MatrixType lType = std::get_if<Typing::MatrixType>(l.get());
        if(std::get_if<Typing::MatrixType>(l.get()) != nullptr){
            if(std::get_if<Typing::MatrixType>(r.get()) == nullptr) return false;
            return
        } else if(std::get_if<Typing::FunctionType>(l.get()) != nullptr){

        } else if(std::get_if<Typing::GenericType>(l.get()) != nullptr){

        }
    }
};

class SymbolTable {
   private:
    // Vector that stores names of variables along with the depth we find them
    std::map<std::string,
             std::map<std::string,
                      SymbolTableEntry>> data;

    std::vector<std::string> functionStack; // Used to identify which function we should be inside

    // Function part separate to variables
    std::map<std::string,
             std::map<std::vector<std::shared_ptr<Typing::Type>>,
                      FunctionTableEntry>> funcTable;

   public:
    // Symbol data
    std::shared_ptr<SymbolTableEntry> getValue(const std::string& symbolName,
                                               const std::string& funcName,
                                               const std::string& funcNamespace = "");

    void setValue(std::shared_ptr<Typing::Type> type, llvm::Value* storeVal, const std::string& symbolName,
                  const std::string& funcName, const std::string& funcNamespace = "");
    void updateValue(llvm::Value* value, const std::string& symbolName,
                     const std::string& funcName, const std::string& funcNamespace = "");

    bool inSymbolTable(const std::string& symbolName, const std::string& funcName, const std::string& funcNamespace = "");

    // Function data
    void addNewFunction(const std::string &funcName,
                        std::vector<std::shared_ptr<Typing::Type>> params,
                        const std::string &funcNamespace = "");
    void setFunctionData(const std::string &funcName,
                         std::vector<std::shared_ptr<Typing::Type>> params,
                         llvm::Function* func,
                         const std::string &funcNamespace = "");
    FunctionTableEntry getFunction(const std::string& funcName,
                                   std::vector<std::shared_ptr<Typing::Type>> params,
                                   const std::string &funcNamespace = "");
    bool isFunctionDefined(const std::string& funcName,
                           const std::string& funcNamespace = "");
    bool isFunctionDefinedParam(const std::string& funcName,
                                const std::vector<std::shared_ptr<Typing::Type>> &params,
                                const std::string &funcNamespace = "");
    // Function stack
    void escapeFunction();
    std::string getCurrentFunction();
};
}  // namespace Utils