#pragma once

#include "ASTNode.hpp"
#include <vector>
#include <string>

namespace AST {
class TypeDefAttributeNode : public Node {
   public:
    std::string name;
    std::shared_ptr<Typing::Type> attrType;
};
}  // namespace AST
