#pragma once

#include <string>
#include <vector>

#include "ASTNode.hpp"

namespace AST {
class TypeDefAttributeNode : public Node {
   public:
    std::string name;
    std::shared_ptr<Typing::Type> attrType;
};
}  // namespace AST
