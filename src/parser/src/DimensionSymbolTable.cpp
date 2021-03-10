#include <span>

#include "DimensionsSymbolTable.hpp"
namespace Analysis {

AST::Node* DimensionSymbolTable::search_impl(const std::string_view name) const noexcept {
    AST::Node* ret{};
    // try searching root namespace if it exists
    if (this->namespaces.contains("")) {
        ret = this->namespaces.at("")->search_impl(name);
    }
    if (ret == nullptr) {
        if (name.size() >= 2) {
            ret = this->namespaces.at(name[0])->search_impl(name.substr(1));
        } else if (name.size() == 1) {
            if (this->values.contains(name[0])) ret = this->values.at(name[0]);
        }
    }
    return ret;
}

}  // namespace Analysis
