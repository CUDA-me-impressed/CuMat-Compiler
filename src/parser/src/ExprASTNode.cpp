#include "ExprASTNode.hpp"

#include <utility>

#include "TreePrint.hpp"

std::shared_ptr<Typing::Type> AST::ExprNode::getType() const { return this->type; }

void AST::ExprNode::setType(std::shared_ptr<Typing::Type> ty) { this->type = std::move(ty); }

std::string AST::ExprNode::toTree(const std::string& prefix, const std::string& childPrefix) const {
    using namespace Tree;
    std::string str{prefix + std::string{"Expression\n"}};
    for (auto const& node : this->children) {
        if (&node != &this->children.back()) {
            str += node->toTree(childPrefix + T, childPrefix + I);
        } else
            str += node->toTree(childPrefix + L, childPrefix + B);
    }
    return str;
}