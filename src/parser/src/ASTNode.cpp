//
// Created by tobyl on 12/11/2020.
//

#include "ASTNode.hpp"

namespace AST {
Node::Node(std::string textRep) { this->literalText = std::move(textRep); }

void Node::addChild(std::shared_ptr<Node> n) { this->children.push_back(std::move(n)); }

std::string Node::toString() const { return this->literalText; }

std::shared_ptr<Typing::Type> Node::getType() const { return this->type; }

void Node::setType(std::shared_ptr<Typing::Type> ty) { this->type = ty; }

void Node::semanticPass() {
    for (auto const& child : this->children) child->semanticPass();
}

llvm::Value* Node::codeGen(Utils::IRContext* context) {
    for (auto const& child : this->children) {
        child->codeGen(context);
    }
    return nullptr;
}

}  // namespace AST
