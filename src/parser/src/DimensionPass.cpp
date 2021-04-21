#include "DimensionPass.hpp"

#include <ASTNode.hpp>
#include <BinaryExprNode.hpp>
#include <DimensionsSymbolTable.hpp>
#include <Type.hpp>
#include <iostream>

#include "Utils.hpp"

bool expandableDimension(const Typing::Type& left, const Typing::Type& right);
std::vector<uint> expandedDimension(const Typing::MatrixType& left, const Typing::MatrixType& right);

void AST::Node::dimensionPass(Analysis::DimensionSymbolTable* nt) {
    for (auto const& child : this->children) child->dimensionPass(nt);
}
void AST::Node::dimensionNamePass(Analysis::DimensionSymbolTable* nt) {
    for (auto const& child : this->children) child->dimensionNamePass(nt);
}

void AST::BinaryExprNode::dimensionPass(Analysis::DimensionSymbolTable* nt) {
    this->lhs->dimensionPass(nt);
    this->rhs->dimensionPass(nt);
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
                auto* t = std::get_if<Typing::MatrixType>(&*this->type);
                if (t) {
                    t->dimensions = expandedDimension(std::get<Typing::MatrixType>(*this->lhs->type),
                                                      std::get<Typing::MatrixType>(*this->rhs->type));
                }
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

void dimension_error(const std::string& message, AST::Node* node) {
    std::cerr << "Dimension Error" << std::endl;
    std::cerr << node->literalText << std::endl;
    std::cerr << message << std::endl;
    throw std::runtime_error{message};
}
