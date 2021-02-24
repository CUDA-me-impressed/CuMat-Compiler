//
// Created by thomas on 24/01/2021.
//
#pragma once

#include <string>

#include "Utils.hpp"

namespace Tree {

const bool WIDE = true;

// Allow us to use a more compact tree if preferred
constexpr const char* T = WIDE ? "├─" : "├";
constexpr const char* I = WIDE ? "│ " : "│";
constexpr const char* L = WIDE ? "└─" : "└";
constexpr const char* B = WIDE ? "  " : " ";

std::string printType(const Typing::Type& type) {
    std::string str =
        std::visit(overloaded{
                       [](const Typing::MatrixType& arg) -> std::string {
                           std::string ret{};
                           switch (arg.primType) {
                               case Typing::PRIMITIVE::STRING:
                                   ret += "String";
                                   break;
                               case Typing::PRIMITIVE::INT:
                                   ret += "Int";
                                   break;
                               case Typing::PRIMITIVE::FLOAT:
                                   ret += "Float";
                                   break;
                               case Typing::PRIMITIVE::BOOL:
                                   ret += "Bool";
                                   break;
                               case Typing::PRIMITIVE::NONE:
                                   ret += "Void";
                                   break;
                               default:
                                   ret += "~InvalidMatType~";
                           }
                           for (const auto& i : arg.dimensions) {
                               if (&i == &arg.dimensions.front()) {
                                   ret += "[";
                               }
                               ret += std::to_string(i);
                               if (&i == &arg.dimensions.back()) {
                                   ret += "]";
                               } else {
                                   ret += ", ";
                               }
                           }
                           return ret;
                       },
                       [](const Typing::FunctionType& arg) -> std::string {
                           std::string ret{};
                           for (const auto& i : arg.parameters) {
                               if (&i == &arg.parameters.front()) {
                                   ret += "(";
                               }
                               ret += printType(*i);
                               if (&i == &arg.parameters.back()) {
                                   ret += ")";
                               } else {
                                   ret += ", ";
                               }
                           }
                           ret += "->";
                           ret += printType(*arg.returnType);
                           return ret;
                       },
                       [](const Typing::GenericType& arg) -> std::string { return std::string{"<"} + arg.name + ">"; },
                   },
                   type);
    return std::move(str);
}
}  // namespace Tree
