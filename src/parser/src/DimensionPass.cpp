
#include <BinaryExprNode.hpp>
#include <DimensionsSymbolTable.hpp>
#include <Type.hpp>

#include "Utils.hpp"

bool expandableDimension(const Typing::Type& left, const Typing::Type& right);

void AST::Node::dimensionPass(Analysis::DimensionSymbolTable* nt) {
    for (auto const& child : this->children) child->dimensionPass(nt);
    for (auto const& child : this->children) child->dimensionPass(nt);
}

void AST::BinaryExprNode::dimensionPass(Analysis::DimensionSymbolTable* nt) {
    switch (this->op) {
        case BIN_OPERATORS::PLUS:
        case BIN_OPERATORS::MINUS:
        case BIN_OPERATORS::MUL:
        case BIN_OPERATORS::DIV:
        case BIN_OPERATORS::LOR:
        case BIN_OPERATORS::LAND:
        case BIN_OPERATORS::LT:
        case BIN_OPERATORS::GT:
        case BIN_OPERATORS::LTE:
        case BIN_OPERATORS::GTE:
        case BIN_OPERATORS::EQ:
        case BIN_OPERATORS::NEQ:
        case BIN_OPERATORS::BAND:
        case BIN_OPERATORS::BOR:
        case BIN_OPERATORS::POW:
            if (expandableDimension(*this->lhs->type, *this->rhs->type)) {
            }
            break;
        case BIN_OPERATORS::MATM:
            // uses different dimensional rules...
            break;
        case BIN_OPERATORS::CHAIN:
            break;
    }
}

bool expandableDimensionMatrix(const Typing::MatrixType& left, const Typing::MatrixType& right) {
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
        if (small->dimensions[i] && big->dimensions[i]) {
        }
    }
    return false;
}

bool expandableDimension(const Typing::Type& left, const Typing::Type& right) {
    if (std::holds_alternative<Typing::MatrixType>(left) && std::holds_alternative<Typing::MatrixType>(right)) {
        return expandableDimensionMatrix(std::get<Typing::MatrixType>(left), std::get<Typing::MatrixType>(right));
    }
    return false;
}

std::vector<uint> expandedDimension(const Typing::MatrixType& left, const Typing::MatrixType& right) {
    const Typing::MatrixType* big;
    if (left.rank > right.rank) {
        big = &left;
    } else {
        big = &right;
    }
    return {big->dimensions};
}