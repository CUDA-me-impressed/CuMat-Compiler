//
// Created by thomas on 23/11/2020.
//

#pragma once

#include <map>
#include <string>

#include "ASTNode.hpp"
namespace AST {

class NameTable {
   protected:
    std::map<std::string, std::unique_ptr<NameTable>> namespaces;
    std::map<std::string, Node*> values;

   public:

};
}  // namespace AST