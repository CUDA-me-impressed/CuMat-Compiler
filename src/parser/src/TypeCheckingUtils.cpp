#include "TypeCheckingUtils.hpp"

#include <iostream>

#include "ExprASTNode.hpp"
#include "Type.hpp"

std::shared_ptr<Typing::Type> TypeCheckUtils::makeMatrixType(const std::vector<uint> dimensions,
                                                             Typing::PRIMITIVE primType) {
    auto ty = Typing::MatrixType();
    ty.dimensions = dimensions;
    ty.rank = dimensions.size();
    ty.primType = primType;
    std::shared_ptr<Typing::Type> type = std::make_shared<Typing::Type>(ty);
    return type;
}

void TypeCheckUtils::assertLogicalType(Typing::PRIMITIVE ty) {
    if (not(ty == Typing::PRIMITIVE::BOOL or ty == Typing::PRIMITIVE::INT)) {
        std::string message = "Expected: bool, int";
        TypeCheckUtils::wrongTypeError(message, ty);
    }
}

void TypeCheckUtils::assertNumericType(Typing::PRIMITIVE ty) {
    if (not(ty == Typing::PRIMITIVE::INT or ty == Typing::PRIMITIVE::FLOAT)) {
        std::string message = "Expected: int, float";
        TypeCheckUtils::wrongTypeError(message, ty);
    }
}

void TypeCheckUtils::assertBooleanType(Typing::PRIMITIVE ty) {
    if (not(ty == Typing::PRIMITIVE::BOOL)) {
        std::string message = "Expected: bool";
        TypeCheckUtils::wrongTypeError(message, ty);
    }
}

std::string TypeCheckUtils::primToString(Typing::PRIMITIVE ty) {
    switch (ty) {
        case Typing::PRIMITIVE::STRING:
            return "string";
        case Typing::PRIMITIVE::INT:
            return "int";
        case Typing::PRIMITIVE::FLOAT:
            return "float";
        case Typing::PRIMITIVE::BOOL:
            return "bool";
        case Typing::PRIMITIVE::NONE:
            return "none";
    }
}

void TypeCheckUtils::wrongTypeError(std::string message, Typing::PRIMITIVE ty) {
    std::cerr << "Wrong type encountered/n" << message << "/n"
              << "Found: " << primToString(ty) << std::endl;
    std::exit(TypeCheckUtils::ErrorCodes::WRONG_TYPE);
}

void TypeCheckUtils::assertMatchingTypes(Typing::PRIMITIVE lhs, Typing::PRIMITIVE rhs) {
    if (not(lhs == rhs)) {
        std::cerr << "Mismatched types: " << primToString(lhs) << ", " << primToString(rhs) << std::endl;
        std::exit(TypeCheckUtils::ErrorCodes::MISMATCH_CODE);
    }
}

void TypeCheckUtils::assertCompatibleTypes(Typing::PRIMITIVE lhs, Typing::PRIMITIVE rhs) {
    bool compatible;
    switch (lhs) {
        case Typing::PRIMITIVE::STRING:
            compatible = rhs == Typing::PRIMITIVE::STRING;
            break;
        case Typing::PRIMITIVE::INT:
        case Typing::PRIMITIVE::FLOAT:
            compatible = (rhs == Typing::PRIMITIVE::INT or rhs == Typing::PRIMITIVE::FLOAT);
            break;
        case Typing::PRIMITIVE::BOOL:
            compatible = (rhs == Typing::PRIMITIVE::INT or rhs == Typing::PRIMITIVE::BOOL);
            break;
        case Typing::PRIMITIVE::NONE:
            compatible = false;
            break;
    }
    if (not compatible) {
        std::cerr << "Incompatible types: " << primToString(lhs) << ", " << primToString(rhs) << std::endl;
        std::exit(TypeCheckUtils::ErrorCodes::MISMATCH_CODE);
    }
}

Typing::MatrixType TypeCheckUtils::extractMatrixType(std::shared_ptr<AST::ExprNode> node) {
    Typing::MatrixType exprType;
    try {
        exprType = std::get<Typing::MatrixType>(*node->type);
    } catch (std::bad_cast b) {
        std::cout << "Caught: " << b.what();
    }
    return exprType;
}