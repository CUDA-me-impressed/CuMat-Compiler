//
// Created by thomas on 23/11/2020.
//

#include "NameTable.hpp"

#include <span>
namespace Analysis {

AST::Node* NameTable::search_impl(
    const std::span<const std::string> name) const noexcept {
    AST::Node* ret{};
    if (this->namespaces.contains("")) {
        ret = this->namespaces.at("")->search_impl(name);
    }
    if (ret == nullptr) {
        if (name.size() >= 2) {
            ret = this->namespaces.at(name[0])->search_impl(name.subspan(1));
        } else if (name.size() == 1) {
            if (this->values.contains(name[0])) ret = this->values.at(name[0]);
        }
    }
    return ret;
}

}  // namespace Analysis
