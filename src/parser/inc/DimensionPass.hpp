//
// Created by thomas on 18/04/2021.
//

#pragma once

#include <string>

#include "ASTNode.hpp"

void dimension_error(const std::string& message, AST::Node* node);
bool expandableDimensionMatrix(const Typing::MatrixType& left, const Typing::MatrixType& right);
bool expandableDimension(const Typing::Type& left, const Typing::Type& right);
std::vector<uint> expandedDimension(const Typing::MatrixType& left, const Typing::MatrixType& right);