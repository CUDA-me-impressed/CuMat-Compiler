#include "Type.hpp"

#include <llvm/IR/Module.h>

#include <iostream>
#include <numeric>
#include <stdexcept>

/**
 * Returns the amount of bits required to store a single element of the
 * primitive type within CuMat
 * @return
 */
int Typing::MatrixType::offset() const {
    switch (primType) {
        case PRIMITIVE::STRING:
        case PRIMITIVE::BOOL:
            return 8;
        case PRIMITIVE::INT:
        case PRIMITIVE::FLOAT:
            return 64;
        case PRIMITIVE::NONE:
            throw std::runtime_error("Invalid type for offset");
    }
}
int Typing::MatrixType::getLength() const {
    return std::accumulate(this->dimensions.begin(), this->dimensions.end(), 1, std::multiplies());
}

llvm::Type* Typing::MatrixType::getLLVMType(llvm::Module* module) const {
    llvm::Type* ty;
    switch (this->primType) {
        case Typing::PRIMITIVE::INT: {
            ty = static_cast<llvm::Type*>(llvm::Type::getInt64Ty(module->getContext()));
            break;
        }
        case Typing::PRIMITIVE::FLOAT: {
            ty = llvm::Type::getFloatTy(module->getContext());
            break;
        }
        case Typing::PRIMITIVE::BOOL: {
            ty = static_cast<llvm::Type*>(llvm::Type::getInt1Ty(module->getContext()));
            break;
        }
        default: {
            std::cerr << "Cannot find a valid type for type" << std::endl;
            // Assign the type to be an integer
            ty = static_cast<llvm::Type*>(llvm::Type::getInt64Ty(module->getContext()));
            break;
        }
        case Typing::PRIMITIVE::STRING:
        case Typing::PRIMITIVE::NONE:
            ty = nullptr;
    }
    return ty;
}
