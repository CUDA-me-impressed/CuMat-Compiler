#include "TypeCheckingSymbolTable.hpp"
#include "TypeCheckingUtils.hpp"

std::shared_ptr<Typing::Type> TypeCheckUtils::TypeCheckingSymbolTable::getType(std::string typeName) {
    if (this->typeData.contains(typeName)) {
        return std::make_shared<Typing::Type>();
    }
    return nullptr;
}

void TypeCheckUtils::TypeCheckingSymbolTable::storeType(std::string typeName, std::shared_ptr<Typing::Type> typePtr) {
    if (!typeData.contains(typeName)) {
        this->typeData[typeName] = std::move(typePtr);
    } else {
        TypeCheckUtils::alreadyDefinedError(typeName);
    }
}

bool TypeCheckUtils::TypeCheckingSymbolTable::inSymbolTable(std::string typeName) {
    return typeData.contains(typeName);
}