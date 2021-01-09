#include "SymbolTable.hpp"

std::shared_ptr<Utils::SymbolTableEntry> Utils::SymbolTable::getValue(const std::string& symbolName,
                                                                      const std::string& funcName,
                                                                      const std::string& funcNamespace) {
    const std::string fullSymbolName = funcNamespace + "::" + symbolName;
    if (data.contains(funcName)) {
        if (data[funcName].contains(fullSymbolName)) {
            return std::make_shared<SymbolTableEntry>(data[funcName][fullSymbolName]);
        } else {
            throw std::runtime_error("Symbol [" + fullSymbolName + "] out of scope");
        }
    } else {
        throw std::runtime_error("Cannot find function [" + funcName + "] for symbol [" + fullSymbolName + "]");
    }
}
void Utils::SymbolTable::setValue(std::shared_ptr<Typing::Type> type, llvm::Value* storeVal,
                                  const std::string& symbolName, const std::string& funcName,
                                  const std::string& funcNamespace) {
    const std::string fullSymbolName = funcNamespace + "::" + symbolName;
    if (!data.contains(funcName)) {
        // let us add an empty map
        this->data[funcName] = std::map<std::string, SymbolTableEntry>();
    }

    // Symbol table does not check if previously added, will override
    this->data[funcName][fullSymbolName] = {type, storeVal};
}
void Utils::SymbolTable::addFunction(const std::string& funcName, const std::string& funcNamespace) {
    this->functionStack.emplace_back(funcName + funcNamespace);
}
void Utils::SymbolTable::escapeFunction() {
    if(this->functionStack.empty())
        throw std::runtime_error("Failed to escape the function within code-block generation. No function!");
    this->functionStack.erase(this->functionStack.end());
}
const std::string& Utils::SymbolTable::getCurrentFunction() { return *this->functionStack.end(); }


bool Utils::SymbolTable::inSymbolTable(const std::string& symbolName, const std::string &funcName, const std::string& funcNamespace) {
    // Check if we store the function name itself first
    if(!this->data.contains(funcName))
        return false;

    // Check if the symbol and its corresponding namespace exists within the symbol table
    std::string fullSymbolName = funcNamespace + "::" + symbolName;
    return this->data[funcName].contains(fullSymbolName);
}

void Utils::SymbolTable::updateValue(llvm::Value* value, const std::string& symbolName,
                                     const std::string& funcName, const std::string& funcNamespace) {
    if(!this->data.contains(funcName)){
        throw std::runtime_error("[Internal Error] Could not update " + symbolName + " to new value function [" +
                                 funcName + "] not found!");
    }

    std::string fullSymbolName = funcNamespace + "::" + symbolName;
    if(!this->data[funcName].contains(fullSymbolName)){
        throw std::runtime_error("[Internal Error] Could not update " + symbolName + " to new value [" +
                                     fullSymbolName + "] not found!");
    }
    this->data[funcName][fullSymbolName] = {this->data[funcName][fullSymbolName].type, value};
}
