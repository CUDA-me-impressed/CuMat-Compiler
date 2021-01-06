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
