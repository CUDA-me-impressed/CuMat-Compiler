#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Type.hpp"

namespace TypeCheckUtils {

    struct VariableEntry {
        bool isFunction;
        std::string nameSpace;
        std::string funcName;
        std::shared_ptr<Typing::Type> varType;
    };

    class TypeCheckingSymbolTable {
       private:
        std::map<std::string, std::map<std::string, std::shared_ptr<Typing::Type>>> funcTypes;
        std::map<std::string, VariableEntry> varTypes;

       public:
        std::shared_ptr<Typing::Type> getFuncType(std::string funcName, std::string nameSpace);
        void storeFuncType(std::string funcName, std::string nameSpace, std::shared_ptr<Typing::Type> typePtr);

        std::shared_ptr<Typing::Type> getVarType(std::string varName);
        void storeVarType(std::string typeName, std::shared_ptr<Typing::Type> typePtr, std::string nameSpace="", std::string funcName="");

        bool inVarTable(std::string typeName);
        bool inFuncTable(std::string funcName, std::string nameSpace);
    };
};
