#include "Type.hpp"

#include <llvm/IR/Module.h>

#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "CodeGenUtils.hpp"

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

const std::vector<uint>& Typing::MatrixType::getDimensions() const { return this->dimensions; }

llvm::Type* Typing::MatrixType::getLLVMPrimitiveType(Utils::IRContext* context) const {
    llvm::Type* ty = nullptr;
    switch (this->primType) {
        case Typing::PRIMITIVE::INT: {
            ty = static_cast<llvm::Type*>(llvm::Type::getInt64Ty(context->module->getContext()));
            break;
        }
        case Typing::PRIMITIVE::FLOAT: {
            ty = llvm::Type::getFloatTy(context->module->getContext());
            break;
        }
        case Typing::PRIMITIVE::BOOL: {
            ty = static_cast<llvm::Type*>(llvm::Type::getInt1Ty(context->module->getContext()));
            break;
        }
        default: {
            std::cerr << "Cannot find a valid type" << std::endl;
            // Assign the type to be an integer
            ty = static_cast<llvm::Type*>(llvm::Type::getInt64Ty(context->module->getContext()));
            break;
        }

            // TODO: Replace with proper types
            // NOLINTNEXTLINE duplicate case
        case Typing::PRIMITIVE::STRING:
            ty = static_cast<llvm::Type*>(llvm::Type::getInt64Ty(context->module->getContext()));
            break;
        case Typing::PRIMITIVE::NONE:
            ty = static_cast<llvm::Type*>(llvm::Type::getInt64Ty(context->module->getContext()));
            break;
    }
    return ty;
}

template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

llvm::Type* Typing::MatrixType::getLLVMType(Utils::IRContext* context) {
    llvm::Type* primTy = this->getLLVMPrimitiveType(context);
    auto rank = this->rank;
    auto length = this->getLength();

    // generate a signature for a matrix
    std::size_t seed = 0;
    hash_combine(seed, rank);
    hash_combine(seed, length);
    hash_combine(seed, primTy);
    // static function members persist through calls, cursed I know
    static std::unordered_map<std::size_t, llvm::Type*> existingStructTypes;
    if (existingStructTypes.contains(seed)) {
        return existingStructTypes.at(seed);
    }

    llvm::ArrayType* matDataType = llvm::ArrayType::get(primTy, this->getLength());
    auto* matDataPtrType = matDataType->getPointerTo();

    auto rankConst = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, this->rank));
    auto numBytes = llvm::ConstantInt::get(context->module->getContext(),
                                           llvm::APInt(64, (this->getLength() * this->offset()) / 8));

    std::vector<llvm::Type*> headerTypes;
    headerTypes.push_back(matDataPtrType);
    headerTypes.push_back(rankConst->getType());  // Rank
    headerTypes.push_back(numBytes->getType());   // # of bytes
    // TODO: Tidy up
    for (int i = 0; i < this->rank; i++) {  // Dimensions
        auto val = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, this->dimensions.at(i)));
        headerTypes.push_back(val->getType());
    }

    auto matHeaderType = llvm::StructType::create(headerTypes);
    matHeaderType->setName("matHeader");

    existingStructTypes.try_emplace(seed, matHeaderType);

    return matHeaderType;
}

Typing::PRIMITIVE Typing::MatrixType::getPrimitiveType() const { return this->primType; }