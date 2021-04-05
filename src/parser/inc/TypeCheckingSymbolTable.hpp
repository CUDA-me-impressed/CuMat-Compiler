#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Type.hpp"

namespace TypeCheckUtils {

    struct variableEntry {
        bool isFunction;
        std::string nameSpace;
        std::string funcName;
        std::shared_ptr<Typing::Type> varType;
    };

    class TypeCheckingSymbolTable {
       private:
        std::map<std::string, std::shared_ptr<Typing::Type>> funcTypes;
        std::map<std::string, variableEntry> varTypes;

       public:
        std::shared_ptr<Typing::Type> getType(std::string typeName);

        void storeType(std::string typeName, std::shared_ptr<Typing::Type> typePtr);

        bool inSymbolTable(std::string typeName);
    };
};
