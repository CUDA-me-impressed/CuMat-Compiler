//
// Created by thomas on 23/11/2020.
//

#pragma once

#include <map>
#include <span>
#include <string>

#include "ASTNode.hpp"

namespace Analysis {

class NameTable {
   private:
    std::map<std::string, std::unique_ptr<NameTable>> namespaces{};
    std::map<std::string, AST::Node*> values{};

   public:
    [[nodiscard]] AST::Node* search_impl(
        std::span<const std::string> name) const noexcept;
};
}  // namespace Analysis
