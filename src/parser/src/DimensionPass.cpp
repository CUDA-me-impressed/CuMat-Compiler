#include "DimensionPass.hpp"

#include <ASTNode.hpp>
#include <DimensionsSymbolTable.hpp>
#include <Type.hpp>
#include <iostream>

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
