//
// Created by tobyl on 14/11/2020.
//

#pragma once

#include <string>
#include <vector>

namespace Typing {
enum class PRIMITIVE { STRING, INT, FLOAT, BOOL, NONE };

class Type {
   public:
    bool isPrimitive;
    bool isGeneric;
    bool isFunction;

    std::string name;
    // If !isPrimitive then NONE
    PRIMITIVE primType;

    int offset();
};

class FunctionType : public Type {
   public:
    Type returnType;
    std::vector<Type> parameters;
};

class MatrixType : public Type {
   public:
    uint rank;  // 1 = Vector, 2 = Matrix, 3 = 3D matrix...
    std::vector<uint> dimensions;
};
}  // namespace Typing
