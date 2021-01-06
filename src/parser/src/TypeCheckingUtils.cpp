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