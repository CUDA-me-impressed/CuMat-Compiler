//
// Created by tobyl on 12/11/2020.
//

#include "ASTNode.hpp"
namespace AST {
Node::Node(std::string textRep) { this->literalText = std::move(textRep); }

void Node::addChild(std::shared_ptr<Node> n) {
    this->children.push_back(std::move(n));
}

std::string Node::toString() const { return this->literalText; }

void Node::semanticPass() {
    for (auto const& child : this->children) child->semanticPass();
}

void Node::codeGen(llvm::Module* TheModule, llvm::IRBuilder<> * Builder, llvm::Function* fp) {
    for (auto const& child : this->children) child->codeGen(TheModule, Builder, fp);
}

}  // namespace AST
