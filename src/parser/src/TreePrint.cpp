#include <Type.hpp>
#include <string>
#include <variant>

#include "Utils.hpp"

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
                       [](const Typing::CustomType& arg) -> std::string { return std::string{"<"} + arg.name + ">"; },
                   },
                   type);
    return std::move(str);
}