//
// Created by thomas on 23/11/2020.
//

#pragma once

#include <map>
#include <span>
#include <string>

#include "ASTNode.hpp"

namespace Analysis {

class DimensionSymbolTable {
   private:
    std::map<std::string, std::unique_ptr<DimensionSymbolTable>> namespaces{};
    std::map<std::string, AST::Node*> values{};

   public:
    [[nodiscard]] AST::Node* search_impl(const std::string_view name) const noexcept;
};
}  // namespace Analysis
