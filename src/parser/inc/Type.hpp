//
// Created by tobyl on 14/11/2020.
//

#pragma once

#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <string>
#include <variant>
#include <vector>

namespace Utils {
struct IRContext;
}

namespace Typing {

enum class PRIMITIVE { STRING, INT, FLOAT, BOOL, NONE };

class MatrixType {
   public:
    uint rank;  // 1 = Vector, 2 = Matrix, 3 = 3D matrix...
    std::vector<uint> dimensions;

    PRIMITIVE primType;

    [[nodiscard]] int getLength() const;
    [[nodiscard]] int offset() const;
    [[nodiscard]] const std::vector<uint>& getDimensions() const { return this->dimensions; }
    bool simpleDimensionCompatible(const MatrixType& val) const { return true; };  // TODO make this not a noop
    llvm::Type* getLLVMType(Utils::IRContext* module) const;
};

class GenericType {
    std::string name;
};

class FunctionType {
   public:
    MatrixType returnType;
    std::vector<std::variant<FunctionType, GenericType, MatrixType>> parameters;

    std::variant<FunctionType, GenericType, MatrixType> resultingType(
        const std::vector<std::variant<FunctionType, GenericType, MatrixType>>& args) {
        return returnType;
    };  // TODO make this not a noop
};

using Type = std::variant<FunctionType, GenericType, MatrixType>;
}  // namespace Typing
