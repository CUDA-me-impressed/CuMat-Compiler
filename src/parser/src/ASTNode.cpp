//
// Created by tobyl on 12/11/2020.
//

#include "ASTNode.hpp"

ASTNode::ASTNode(std::shared_ptr<ASTNode> creator, std::string textRep) {
    this->parent = std::move(creator);
    this->literalText = std::move(textRep);
}
