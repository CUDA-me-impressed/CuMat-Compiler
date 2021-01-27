//
// Created by tobyl on 12/11/2020.
//

#include "ASTNode.hpp"

#include "TreePrint.hpp"

namespace AST {
Node::Node(std::string textRep) { this->literalText = std::move(textRep); }

void Node::addChild(std::shared_ptr<Node> n) { this->children.push_back(std::move(n)); }

std::string Node::toString() const { return this->literalText; }

void Node::semanticPass() {
    for (auto const& child : this->children) child->semanticPass();
}

llvm::Value* Node::codeGen(Utils::IRContext* context) {
    for (auto const& child : this->children) {
        child->codeGen(context);
    }
    return nullptr;
}

std::string Node::toTree(const std::string& prefix, const std::string& childPrefix) const {
    using namespace Tree;
    std::string str{prefix + std::string{"Node\n"}};
    for (auto const& node : this->children) {
        if (&node != &this->children.back()) {
            str += node->toTree(childPrefix + T, childPrefix + I);
        } else
            str += node->toTree(childPrefix + L, childPrefix + B);
    }
    return str;
}

}  // namespace AST
