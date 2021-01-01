//
// Created by lloyd on 30/12/2020.
//

#include "TypeCheckingUtils.hpp"
#include "Type.hpp"

std::shared_ptr<Typing::Type> makeGenericType(std::string typeString) {
    auto ty = std::make_shared<Typing::GenericType>();
    ty->name = ty;
    std::shared_ptr<Typing::Type> type = static_pointer_cast<Typing::Type>(ty);
    return type;
}