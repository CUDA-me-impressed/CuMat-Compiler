#pragma once

#include "ExprASTNode.hpp"
#include "Type.hpp"

namespace TypeCheckUtils {

enum ErrorCodes { WRONG_TYPE = 2, MISMATCH_CODE = 3 };

std::shared_ptr<Typing::Type> makeMatrixType(const std::vector<uint> dimensions, Typing::PRIMITIVE primType);

void assertLogicalType(Typing::PRIMITIVE ty);
void assertNumericType(Typing::PRIMITIVE ty);
void assertBooleanType(Typing::PRIMITIVE ty);
void assertMatchingTypes(Typing::PRIMITIVE lhs, Typing::PRIMITIVE rhs);
void assertCompatibleTypes(Typing::PRIMITIVE lhs, Typing::PRIMITIVE rhs);

std::string primToString(Typing::PRIMITIVE ty);

void wrongTypeError(std::string message, Typing::PRIMITIVE ty);

std::string primToString(Typing::PRIMITIVE ty);

Typing::MatrixType extractMatrixType(std::shared_ptr<AST::ExprNode> node);
}  // namespace TypeCheckUtils