//
// Created by tobyl on 14/11/2020.
//

#pragma once

#include <string>
#include <vector>

namespace Typing {
enum class PRIMITIVE { STRING, INT, FLOAT, NONE };

class Type {
    bool isPrimitive;
    bool isGeneric;
    bool isFunction;

    std::string name;
};

class FunctionType : Type {
    Type returnType;
    std::vector<Type> parameters;
};

class MatrixType : Type {
    uint rank;  // 1 = Vector, 2 = Matrix, 3 = 3D matrix...
    std::vector<uint> dimensions;
};

class PrimitiveType : Type {
    PRIMITIVE prim;
};
}  // namespace Typing

