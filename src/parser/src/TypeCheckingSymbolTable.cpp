#include "TypeCheckingSymbolTable.hpp"
#include "TypeCheckingUtils.hpp"

std::shared_ptr<Typing::Type> TypeCheckUtils::TypeCheckingSymbolTable::getFuncType(std::string funcName, std::string nameSpace) {
    if (this->funcTypes.contains(nameSpace)) {
        if (this->funcTypes[nameSpace].contains(funcName)) {
            return funcTypes[nameSpace][funcName];
        }
    }
    TypeCheckUtils::notDefinedError(funcName);
    return nullptr;
}

void TypeCheckUtils::TypeCheckingSymbolTable::storeFuncType(std::string funcName, std::string nameSpace, std::shared_ptr<Typing::Type> typePtr) {
    if (this->inFuncTable(funcName, nameSpace)) {
        TypeCheckUtils::alreadyDefinedError(funcName);
    }
    if (!this->funcTypes.contains(nameSpace)) {
        this->funcTypes[nameSpace] = std::map<std::string, std::shared_ptr<Typing::Type>>();
    }
    this->funcTypes[nameSpace][funcName] = std::move(typePtr);
}

std::shared_ptr<Typing::Type> TypeCheckUtils::TypeCheckingSymbolTable::getVarType(std::string varName) {
    if (this->varTypes.contains(varName)) {
        TypeCheckUtils::VariableEntry entry = this->varTypes[varName];
        if (entry.isFunction) {
            return this->getFuncType(entry.funcName, entry.nameSpace);
        } else {
            return entry.varType;
        }
    }
    TypeCheckUtils::notDefinedError(varName);
    return nullptr;
}

void TypeCheckUtils::TypeCheckingSymbolTable::storeVarType(std::string typeName, std::shared_ptr<Typing::Type> typePtr, std::string nameSpace, std::string funcName) {
    if (this->inVarTable(typeName)) {
        TypeCheckUtils::alreadyDefinedError(typeName);
    }
    TypeCheckUtils::VariableEntry entry;
    entry.funcName = funcName;
    entry.nameSpace = nameSpace;
    entry.varType = std::move(typePtr);
    if (funcName != "") {
        if (this->inFuncTable(funcName, nameSpace)) {
            entry.isFunction = true;
        } else {
            TypeCheckUtils::notDefinedError(funcName);
        }
    } else {
        entry.isFunction = false;
    }
    this->varTypes[typeName] = entry;
}

void TypeCheckUtils::TypeCheckingSymbolTable::removeVarEntry(std::string typeName) {
    if (this->inVarTable(typeName)) {
        this->varTypes.erase(typeName);
    } else {
        TypeCheckUtils::notDefinedError(typeName);
    }
}

bool TypeCheckUtils::TypeCheckingSymbolTable::inVarTable(std::string typeName) {
    return this->varTypes.contains(typeName);
}
bool TypeCheckUtils::TypeCheckingSymbolTable::inFuncTable(std::string funcName, std::string nameSpace) {
    if (this->funcTypes.contains(nameSpace)) {
        return this->funcTypes[nameSpace].contains(funcName);
    } else {
        return false;
    }
}