#include "ExprASTNode.hpp"

#include "TreePrint.hpp"

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