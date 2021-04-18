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
            ty = llvm::Type::getDoubleTy(context->module->getContext());
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
    auto offset = this->offset();

    // generate a signature for a matrix
    std::size_t seed = 0;
    hash_combine(seed, primTy);
    // static function members persist through calls, cursed I know
    if (existingStructTypes.contains(seed)) {
        return existingStructTypes.at(seed);
    }

    auto matHeaderType = getMatHeaderType(context, primTy, rank, length, offset);

    existingStructTypes.try_emplace(seed, matHeaderType);

    return matHeaderType;
}

Typing::PRIMITIVE Typing::MatrixType::getPrimitiveType() const { return this->primType; }

llvm::Type* Typing::MatrixType::getMatHeaderType(Utils::IRContext* context, llvm::Type* primTy, uint rank, int length, int offset) {

    llvm::ArrayType* matDataType = llvm::ArrayType::get(primTy, 0);
    auto* matDataPtrType = matDataType->getPointerTo();

    auto rankConst = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, rank));
    auto numBytes = llvm::ConstantInt::get(context->module->getContext(),
                                           llvm::APInt(64, (length* offset) / 8));

    std::vector<llvm::Type*> headerTypes;
    headerTypes.push_back(matDataPtrType);
    headerTypes.push_back(rankConst->getType());  // Rank
    headerTypes.push_back(numBytes->getType());   // # of bytes

    llvm::ArrayType* matDimensionArr = llvm::ArrayType::get(llvm::Type::getInt64Ty(context->module->getContext()), 0);
    auto* matDimensionPtrType = matDimensionArr->getPointerTo();
    headerTypes.push_back(matDataPtrType);
//    // TODO: Tidy up
//    for (int i = 0; i < this->rank; i++) {  // Dimensions
//        auto val = llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(64, this->dimensions.at(i)));
//        headerTypes.push_back(val->getType());
//    }

    auto matHeaderType = llvm::StructType::create(headerTypes);
    switch (this->primType) {
        case Typing::PRIMITIVE::INT: {
            matHeaderType->setName("matHeaderI");
            break;
        }
        case Typing::PRIMITIVE::FLOAT: {
            matHeaderType->setName("matHeaderF");
            break;
        }
    }
    return matHeaderType;
}
