#include "Type.hpp"

#include <stdexcept>

/**
 * Returns the amount of bits required to store a single element of the
 * primitive type within CuMat
 * @return
 */
int Typing::Type::offset() {
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