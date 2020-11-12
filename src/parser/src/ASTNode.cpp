//
// Created by tobyl on 12/11/2020.
//

#include "ASTNode.hpp"

ASTNode::ASTNode(std::string textRep) {
    this->parent = nullptr;
    this->literalText = std::move(textRep);
}

void ASTNode::addChild(std::shared_ptr<ASTNode> n) {
    n->parent = this;
    this->children.push_back(std::move(n));
}

std::string ASTNode::toString() { return this->literalText; }
