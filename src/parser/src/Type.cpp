#include "Type.hpp"

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
            break;
        case PRIMITIVE::INT:
        case PRIMITIVE::FLOAT:
            return 64;
            break;
    }
}