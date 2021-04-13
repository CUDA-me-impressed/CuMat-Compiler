//
// Created by thomas on 24/01/2021.
//
#pragma once

#include <string>

namespace Tree {

const bool WIDE = true;

// Allow us to use a more compact tree if preferred
constexpr const char* T = WIDE ? "├─" : "├";
constexpr const char* I = WIDE ? "│ " : "│";
constexpr const char* L = WIDE ? "└─" : "└";
constexpr const char* B = WIDE ? "  " : " ";

std::string printType(const Typing::Type& type);
}  // namespace Tree
