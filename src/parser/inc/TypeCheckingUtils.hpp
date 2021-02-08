//
// Created by lloyd on 30/12/2020.
//

#ifndef CUMAT_COMPILER_TYPECHECKINGUTILS_HPP
#define CUMAT_COMPILER_TYPECHECKINGUTILS_HPP

#endif  // CUMAT_COMPILER_TYPECHECKINGUTILS_HPP

#include "Type.hpp"
#include "ExprASTNode.hpp"

int wrongTypeCode = 2;
int typeMismatchCode = 3;

std::shared_ptr<Typing::Type> makeMatrixType(const std::vector<uint> dimensions, Typing::PRIMITIVE primType);

void assertLogicalType(Typing::PRIMITIVE ty);
void assertNumericType(Typing::PRIMITIVE ty);
void assertBooleanType(Typing::PRIMITIVE ty);
void assertMatchingTypes(Typing::PRIMITIVE lhs, Typing::PRIMITIVE rhs);

std::string primToString(Typing::PRIMITIVE ty);

void wrongTypeError(std::string message, Typing::PRIMITIVE ty);

Typing::MatrixType extractMatrixType(std::shared_ptr<AST::ExprNode> node);
