//
// Created by thomas on 23/11/2020.
//

#pragma once

#include <map>
#include <string_view>
#include <string>

#include "ASTNode.hpp"

namespace Analysis {

class DimensionSymbolTable {
   private:
    std::map<std::string, DimensionSymbolTable*, std::less<>> namespaces{};
    std::map<std::string, AST::Node*, std::less<>> values{};

   public:
    [[nodiscard]] AST::Node* search_impl(std::string_view name) noexcept;
    [[nodiscard]] std::unique_ptr<DimensionSymbolTable> push_scope(const std::string& name = "") {
        auto d = std::make_unique<DimensionSymbolTable>();
        d->namespaces.insert({name, this});
        return std::move(d);
    }
};
}  // namespace Analysis
