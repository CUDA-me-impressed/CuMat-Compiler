#pragma once

#include <vector>
#include <map>
#include <utility>
#include <string>

#include <llvm/IR/Value.h>

namespace Utils {
    class SymbolTable {
       private:
        // Vector that stores names of variables along with the depth we find them
        std::vector<std::map<std::string, llvm::Value*>> data;
        std::map<std::string, std::vector<int>> variableLocations;

       public:
        void newScope();
        void exitScope();

        llvm::Value* getValue(const std::string &symbolName);
        void setValue(const std::string &symbolName, llvm::Value* storeVal);

        bool inScope(const std::string &symbolName);

    };
}