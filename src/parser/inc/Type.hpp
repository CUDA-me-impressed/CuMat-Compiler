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

class CustomType;

class FunctionType;

using Type = std::variant<FunctionType, GenericType, CustomType, MatrixType>;

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

    llvm::Type* getLLVMType(Utils::IRContext* context);

    llvm::Type* getLLVMPrimitiveType(Utils::IRContext* context) const;
    [[nodiscard]] PRIMITIVE getPrimitiveType() const;
};

class GenericType {
   public:
    std::string name;
};

class CustomType {
   public:
    std::string name;
    std::vector<std::pair<std::string, std::shared_ptr<Typing::Type>>> attributes;
};

class FunctionType {
   public:
    std::shared_ptr<Type> returnType;
    std::vector<std::shared_ptr<Type>> parameters;
};
}  // namespace Typing
