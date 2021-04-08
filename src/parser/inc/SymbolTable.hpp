#pragma once

#include <llvm/IR/Value.h>

#include <map>
#include <string>
#include <utility>
#include <vector>
#include "BinaryExprNode.hpp"

#include "Type.hpp"

namespace Utils {

struct SymbolTableEntry {
    std::shared_ptr<Typing::Type> type;
    llvm::Value* llvmVal;
};

struct FunctionTableEntry {
    llvm::Function* func;
};

struct CUDAFunctions {
    llvm::Function* binInt;
    llvm::Function* binFloat;
    llvm::Function* unaryInt;
    llvm::Function* unaryFloat;

};

struct FunctionParamCompare {
    [[nodiscard]] bool equalType(const std::shared_ptr<Typing::Type>& l, const std::shared_ptr<Typing::Type>& r) const {
        bool retVal = false;
        if (std::get_if<Typing::MatrixType>(l.get()) != nullptr) {
            // Matrix type checking
            retVal =
                std::get_if<Typing::MatrixType>(r.get()) != nullptr &&
                std::get_if<Typing::MatrixType>(l.get())->primType < std::get_if<Typing::MatrixType>(r.get())->primType;
        } else if (std::get_if<Typing::FunctionType>(l.get()) != nullptr) {
            // Function type checks (for functions supplied as arguments)
            retVal = std::get_if<Typing::FunctionType>(r.get()) != nullptr &&
                     equalType(std::get_if<Typing::FunctionType>(l.get())->returnType,
                               std::get_if<Typing::FunctionType>(r.get())->returnType);
        } else if (std::get_if<Typing::GenericType>(l.get()) != nullptr) {
            // Generic type checking
            retVal = std::get_if<Typing::GenericType>(r.get()) == nullptr;
        }
        return retVal;
    }

    bool operator()(const std::vector<std::shared_ptr<Typing::Type>>& l,
                    const std::vector<std::shared_ptr<Typing::Type>>& r) const {
        // This is the worst possible way to do this but I really really really am out of options here fuck it
        if (l.size() != r.size()) {
            return l.size() < r.size();  // If not even the same size, use the memory addresses
        }

        bool retVal = true;  // Optimistic assumption
        for (int i = 0; i < l.size(); i++) {
            retVal &= equalType(l.at(i), r.at(i));
        }
        return !retVal;
    }
};

class SymbolTable {
   private:
    // Vector that stores names of variables along with the depth we find them
    std::map<std::string, std::map<std::string, SymbolTableEntry>> data;

    std::vector<std::string> functionStack;  // Used to identify which function we should be inside

    // Function part separate to variables
    std::map<std::string,
             std::map<std::vector<std::shared_ptr<Typing::Type>>, FunctionTableEntry, FunctionParamCompare>>
        funcTable;

    llvm::NamedMDNode* nvvmMetadataNode = nullptr;

    std::map<AST::BIN_OPERATORS, llvm::Function*> binaryCuDAFunctionsInt;
    std::map<AST::BIN_OPERATORS, llvm::Function*> binaryCuDAFunctionsFloat;

   public:
    // Symbol data
    std::shared_ptr<SymbolTableEntry> getValue(const std::string& symbolName, const std::string& funcName,
                                               const std::string& funcNamespace = "");

    void setValue(std::shared_ptr<Typing::Type> type, llvm::Value* storeVal, const std::string& symbolName,
                  const std::string& funcName, const std::string& funcNamespace = "");

    void updateValue(llvm::Value* value, const std::string& symbolName, const std::string& funcName,
                     const std::string& funcNamespace = "");

    bool inSymbolTable(const std::string& symbolName, const std::string& funcName,
                       const std::string& funcNamespace = "");

    // Function data
    void addNewFunction(const std::string& funcName, const std::vector<std::shared_ptr<Typing::Type>>& params,
                        const std::string& funcNamespace = "");

    void setFunctionData(const std::string& funcName, const std::vector<std::shared_ptr<Typing::Type>>& params,
                         llvm::Function* func, const std::string& funcNamespace = "");

    FunctionTableEntry getFunction(const std::string& funcName,
                                   const std::vector<std::shared_ptr<Typing::Type>>& params,
                                   const std::string& funcNamespace = "");

    bool isFunctionDefined(const std::string& funcName, const std::string& funcNamespace = "");

    bool isFunctionDefinedParam(const std::string& funcName, const std::vector<std::shared_ptr<Typing::Type>>& params,
                                const std::string& funcNamespace = "");

    void createNVVMMetadata(Utils::IRContext* context);

    llvm::NamedMDNode* getNVVMMetadata();

    void

    // Function stack
    void escapeFunction();
    void enterFunction(const std::string& function, const std::string& funcNamespace = "");
    std::string getCurrentFunction();
};
}  // namespace Utils