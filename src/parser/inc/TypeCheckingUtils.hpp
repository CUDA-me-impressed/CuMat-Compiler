#pragma once

#include "ExprASTNode.hpp"
#include "Type.hpp"

namespace TypeCheckUtils {

    enum ErrorCodes {
        WRONG_TYPE = 2, MISMATCH_CODE = 3, CASTING_ERROR = 4, NONE_ERROR = 5
    };

    std::shared_ptr<Typing::Type> makeMatrixType(const std::vector<uint> dimensions, Typing::PRIMITIVE primType);
    std::shared_ptr<Typing::Type> makeCustomType(std::string name, const std::vector<std::pair<std::string, std::shared_ptr<Typing::Type>>> attrs);

    bool isBool(Typing::PRIMITIVE ty);
    bool isInt(Typing::PRIMITIVE ty);
    bool isFloat(Typing::PRIMITIVE ty);
    bool isString(Typing::PRIMITIVE ty);
    bool isNone(Typing::PRIMITIVE ty);

    void assertLogicalType(Typing::PRIMITIVE ty);
    void assertNumericType(Typing::PRIMITIVE ty);
    void assertBooleanType(Typing::PRIMITIVE ty);
    void assertMatchingTypes(Typing::PRIMITIVE lhs, Typing::PRIMITIVE rhs);
    void assertCompatibleTypes(Typing::PRIMITIVE lhs, Typing::PRIMITIVE rhs);

    std::string primToString(Typing::PRIMITIVE ty);

    void wrongTypeError(std::string message, Typing::PRIMITIVE ty);
    void castingError();
    void noneError();

    std::string primToString(Typing::PRIMITIVE ty);

    Typing::MatrixType extractMatrixType(std::shared_ptr<AST::ExprNode> node);
    Typing::PRIMITIVE getHighestType(Typing::PRIMITIVE lhs, Typing::PRIMITIVE rhs);
}
