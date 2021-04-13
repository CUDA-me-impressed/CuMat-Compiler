#pragma once

#include <string>
#include <vector>

#include "ASTNode.hpp"
#include "Type.hpp"
#include "TypeDefAttributeNode.hpp"

namespace AST {
class CustomTypeDefNode : public Node {
   public:
    std::string name;
    std::vector<std::shared_ptr<AST::TypeDefAttributeNode>> attributes;

    void semanticPass(Utils::IRContext* context) override;
};
}  // namespace AST
