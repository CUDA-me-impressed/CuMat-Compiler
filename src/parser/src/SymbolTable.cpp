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

void Utils::SymbolTable::escapeFunction() {
    if (this->functionStack.empty())
        throw std::runtime_error("Failed to escape the function within code-block generation. No function!");
    this->functionStack.erase(this->functionStack.end());
}
std::string Utils::SymbolTable::getCurrentFunction() { return this->functionStack.at(this->functionStack.size() - 1); }

bool Utils::SymbolTable::inSymbolTable(const std::string& symbolName, const std::string& funcName,
                                       const std::string& funcNamespace) {
    // Check if we store the function name itself first
    if (!this->data.contains(funcName)) return false;

    // Check if the symbol and its corresponding namespace exists within the symbol table
    std::string fullSymbolName = funcNamespace + "::" + symbolName;
    return this->data[funcName].contains(fullSymbolName);
}

void Utils::SymbolTable::updateValue(llvm::Value* value, const std::string& symbolName, const std::string& funcName,
                                     const std::string& funcNamespace) {
    if (!this->data.contains(funcName)) {
        throw std::runtime_error("[Internal Error] Could not update " + symbolName + " to new value function [" +
                                 funcName + "] not found!");
    }

    std::string fullSymbolName = funcNamespace + "::" + symbolName;
    if (!this->data[funcName].contains(fullSymbolName)) {
        throw std::runtime_error("[Internal Error] Could not update " + symbolName + " to new value [" +
                                 fullSymbolName + "] not found!");
    }
    this->data[funcName][fullSymbolName] = {this->data[funcName][fullSymbolName].type, value};
}

/**
 * Returns true if the function is defined within the namespace
 * @param funcName
 * @param funcNamespace
 * @return
 */
bool Utils::SymbolTable::isFunctionDefined(const std::string& funcName, const std::string& funcNamespace) {
    return this->funcTable.contains(funcNamespace + "::" + funcName);
}

/**
 * Sets the value of the function to be pointing to the llvm::Function object
 * @param funcName
 * @param params
 * @param func
 */
void Utils::SymbolTable::setFunctionData(const std::string& funcName, std::vector<std::shared_ptr<Typing::Type>> params,
                                         llvm::Function* func, const std::string& funcNamespace) {
    // Safety checks
    if (!isFunctionDefined(funcName, funcNamespace)) {
        throw std::runtime_error("Function [" + funcName + "] not defined!");
    }
    const std::string fullFuncName = funcNamespace + "::" + funcName;
    if (!this->funcTable[fullFuncName].contains(params)) {
        // TODO: Either done in typing or made to actually report type issues
        throw std::runtime_error("Function [" + fullFuncName + "] expected different parameters");
    }

    this->funcTable[fullFuncName][params] = {func};
}

/**
 * Returns true if the function with the specified parameters is defined
 * @param funcName
 * @param params
 * @return
 */
bool Utils::SymbolTable::isFunctionDefinedParam(const std::string& funcName,
                                                const std::vector<std::shared_ptr<Typing::Type>>& params,
                                                const std::string& funcNamespace) {
    const std::string fullFuncName = funcNamespace + "::" + funcName;
    if (!isFunctionDefined(funcName)) {
        throw std::runtime_error("Function [" + funcName + "] not defined!");
    }

    return this->funcTable[fullFuncName].contains(params);
}

/**
 * Retrieves the function corresponding to name and type
 * @param funcName
 * @param params
 * @return
 */
Utils::FunctionTableEntry Utils::SymbolTable::getFunction(const std::string& funcName,
                                                          std::vector<std::shared_ptr<Typing::Type>> params,
                                                          const std::string& funcNamespace) {
    const std::string fullFuncName = funcNamespace + "::" + funcName;
    if (!isFunctionDefinedParam(funcName, params, funcNamespace)) {
        throw std::runtime_error("[Internal Error] Cannot retrieve function, not defined");
    }
    return this->funcTable[fullFuncName][params];
}

/**
 * Adds a function within the symbol table and creates a function stack
 * @param funcName
 * @param params
 * @param funcNamespace
 */
void Utils::SymbolTable::addNewFunction(const std::string& funcName, std::vector<std::shared_ptr<Typing::Type>> params,
                                        const std::string& funcNamespace) {
    const std::string fullFuncName = funcNamespace + "::" + funcName;
    this->functionStack.emplace_back(fullFuncName);
    // If we have no override
    if (!this->isFunctionDefined(funcName, funcNamespace)) {
        this->funcTable[fullFuncName] = {};
    }

    this->funcTable[fullFuncName][params] = {};
}
