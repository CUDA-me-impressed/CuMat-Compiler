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

namespace Typing {

enum class PRIMITIVE { STRING, INT, FLOAT, BOOL, NONE };

class MatrixType;
class GenericType;
class FunctionType;

using Type = std::variant<FunctionType, GenericType, MatrixType>;

class MatrixType {
   public:
    uint rank;  // 1 = Vector, 2 = Matrix, 3 = 3D matrix...
    std::vector<uint> dimensions;

    PRIMITIVE primType;

    int getLength();
    int offset();
    llvm::Type* getLLVMType(llvm::Module* module);
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
