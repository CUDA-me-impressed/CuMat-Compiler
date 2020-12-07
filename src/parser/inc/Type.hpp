//
// Created by tobyl on 14/11/2020.
//

#pragma once

#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace Utils {
struct IRContext;
}

namespace Typing {

enum class PRIMITIVE { STRING, INT, FLOAT, BOOL, NONE };

class MatrixType;
class GenericType;
class FunctionType;

using Type = std::variant<FunctionType, GenericType, MatrixType>;

class MatrixType {
   public:
    uint rank{0};  // 1 = Vector, 2 = Matrix, 3 = 3D matrix...
    std::vector<uint> dimensions{};

    PRIMITIVE primType{PRIMITIVE::NONE};

    [[nodiscard]] int getLength() const;
    [[nodiscard]] int offset() const;
    [[nodiscard]] const std::vector<uint>& getDimensions() const;

    // TODO make this not a noop
    [[nodiscard]] bool simpleDimensionCompatible(const MatrixType& val) const { return true; };

    llvm::Type* getLLVMType(Utils::IRContext* context) const;
};

class GenericType {
    std::string name;
};

class FunctionType {
   public:
    std::shared_ptr<Type> returnType;
    std::vector<std::shared_ptr<Type>> parameters;
};
}  // namespace Typing
