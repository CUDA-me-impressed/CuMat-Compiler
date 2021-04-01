#pragma once

#include <string>
#include <vector>

#include "ASTNode.hpp"
#include "TypeDefAttributeNode.hpp"
#include "Type.hpp"

namespace AST {
class CustomTypeDefNode : public Node {
   public:
    std::string name;
    std::vector<std::shared_ptr<AST::TypeDefAttributeNode>> attributes;

    void semanticPass(Utils::IRContext* context) override;
};
}  // namespace AST
