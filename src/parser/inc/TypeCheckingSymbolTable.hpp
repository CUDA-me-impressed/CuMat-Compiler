#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Type.hpp"

namespace TypeCheckUtils {
    class TypeCheckingSymbolTable {
       private:
        std::map<std::string, std::shared_ptr<Typing::Type>> typeData;

       public:
        std::shared_ptr<Typing::Type> getType(std::string typeName);

        void storeType(std::string typeName, std::shared_ptr<Typing::Type> typePtr);

        bool inSymbolTable(std::string typeName);
    };
};
