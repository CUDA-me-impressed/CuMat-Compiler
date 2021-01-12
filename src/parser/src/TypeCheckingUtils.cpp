//
// Created by lloyd on 30/12/2020.
//

#include "TypeCheckingUtils.hpp"

#include "Type.hpp"

std::shared_ptr<Typing::Type> makeGenericType(std::string typeString) {
    auto ty = Typing::GenericType();
    ty.name = typeString;
    std::shared_ptr<Typing::Type> type = std::make_shared<Typing::Type>(ty);
    return type;
}

std::shared_ptr<Typing::Type> makeMatrixType(std::vector<uint> dimensions) {
    auto ty = Typing::MatrixType();
    ty.dimensions = dimensions;
    ty.rank = dimensions.size();
//    ty.primType = ;
    std::shared_ptr<Typing::Type> type = std::make_shared<Typing::Type>(ty);
    return type;
}