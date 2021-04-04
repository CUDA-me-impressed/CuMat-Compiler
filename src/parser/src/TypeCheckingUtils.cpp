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

std::shared_ptr<Typing::Type> TypeCheckUtils::makeCustomType(std::string name, const std::vector<std::pair<std::string, std::shared_ptr<Typing::Type>>> attrs) {
    auto ty = Typing::CustomType();
    ty.name = name;
    ty.attributes = attrs;
    std::shared_ptr<Typing::Type> type = std::make_shared<Typing::Type>(ty);
    return type;
}

std::shared_ptr<Typing::Type> TypeCheckUtils::makeFunctionType(std::shared_ptr<Typing::Type> returnType, const std::vector<std::shared_ptr<Typing::Type>> params) {
    auto ty = Typing::FunctionType();
    ty.returnType = returnType;
    ty.parameters = params;
    std::shared_ptr<Typing::Type> type = std::make_shared<Typing::Type>(ty);
    return type;
}

bool TypeCheckUtils::isBool(Typing::PRIMITIVE ty) {
    return ty == Typing::PRIMITIVE::BOOL;
}

bool TypeCheckUtils::isInt(Typing::PRIMITIVE ty) {
    return ty == Typing::PRIMITIVE::INT;
}

bool TypeCheckUtils::isFloat(Typing::PRIMITIVE ty) {
    return ty == Typing::PRIMITIVE::FLOAT;
}

bool TypeCheckUtils::isString(Typing::PRIMITIVE ty) {
    return ty == Typing::PRIMITIVE::STRING;
}

bool TypeCheckUtils::isNone(Typing::PRIMITIVE ty) {
    return ty == Typing::PRIMITIVE::NONE;
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

void TypeCheckUtils::castingError() {
    std::cerr << "Cannot cast to incompatible type" << std::endl;
    std::exit(TypeCheckUtils::ErrorCodes::WRONG_TYPE);
}

void TypeCheckUtils::noneError() {
    std::cerr << "Type should not be NONE" << std::endl;
    std::exit(TypeCheckUtils::ErrorCodes::NONE_ERROR);
}

void TypeCheckUtils::alreadyDefinedError(std::string funcName) {
    std::cerr << "Function with same name already defined: " << funcName << std::endl;
    std::exit(TypeCheckUtils::ErrorCodes::ALREADY_DEFINED_ERROR);
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
            compatible = (rhs == Typing::PRIMITIVE::INT or rhs == Typing::PRIMITIVE::FLOAT or rhs == Typing::PRIMITIVE::BOOL);
            break;
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

Typing::PRIMITIVE TypeCheckUtils::getHighestType(Typing::PRIMITIVE lhs, Typing::PRIMITIVE rhs) {
    if (lhs == rhs) {
        return lhs;
    }
    switch (lhs) {
        case Typing::PRIMITIVE::INT:
            switch (rhs) {
                case Typing::PRIMITIVE::BOOL:
                    return Typing::PRIMITIVE::INT;

                case Typing::PRIMITIVE::FLOAT:
                    return Typing::PRIMITIVE::FLOAT;

                default:
                    TypeCheckUtils::castingError();
            }

        case Typing::PRIMITIVE::FLOAT:
            if (not (rhs == Typing::PRIMITIVE::INT or rhs == Typing::PRIMITIVE::FLOAT)) {
                TypeCheckUtils::castingError();
            }
            return Typing::PRIMITIVE::FLOAT;

        case Typing::PRIMITIVE::BOOL:
            if (rhs == Typing::PRIMITIVE::INT) {
                return Typing::PRIMITIVE::INT;
            } else {
                TypeCheckUtils::castingError();
            }

        case Typing::PRIMITIVE::STRING:
            TypeCheckUtils::castingError();

        case Typing::PRIMITIVE::NONE:
            TypeCheckUtils::noneError();

    }
}