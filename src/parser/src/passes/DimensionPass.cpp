//
// Created by thomas on 22/11/2020.
//
#import "ASTNode.hpp"
#import "BinaryExprASTNode.hpp"
#import "FunctionExprASTNode.hpp"
#import "LiteralASTNode.hpp"
#import "MatrixASTNode.hpp"
#import "TernaryExprASTNode.hpp"
#import "UnaryExprASTNode.hpp"

namespace AST {

bool expandableDimension(const Typing::Type& left, const Typing::Type& right);

void Node::dimensionPass(Analysis::NameTable* nt) {
    for (auto const& child : this->children) child->dimensionPass(nt);
    for (auto const& child : this->children) child->dimensionPass(nt);
}

void BinaryExprASTNode::dimensionPass(Analysis::NameTable* nt) {
    switch (this->op) {
        case PLUS:
        case MINUS:
        case MUL:
        case DIV:
        case LOR:
        case LAND:
        case LT:
        case GT:
        case LTE:
        case GTE:
        case EQ:
        case NEQ:
        case BAND:
        case BOR:
        case POW:
            if (expandableDimension(*this->lhs->type, *this->rhs->type)) {
            }
            break;
        case MATM:
            // uses different dimensional rules...
            break;
        case CHAIN:
            break;
    }
}

bool expandableDimension(const Typing::MatrixType& left,
                         const Typing::MatrixType& right) {
    const Typing::MatrixType* small;
    const Typing::MatrixType* big;
    if (left.rank > right.rank) {
        small = &right;
        big = &left;
    } else {
        small = &left;
        big = &right;
    }
    for (int i = 0; i < small->rank; i++) {
        if (std::holds_alternative<uint>(small->dimensions[i]) &&
            std::holds_alternative<uint>(big->dimensions[i])) {
        }
    }
    return false;
}

}  // namespace AST