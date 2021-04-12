#include <span>

#include "DimensionsSymbolTable.hpp"
std::vector<std::string_view> Split(std::string_view str, char delim = '.');

namespace Analysis {

AST::Node* DimensionSymbolTable::search_impl(const std::string_view name) noexcept {
    AST::Node* ret{};
    // try searching root namespace if it exists
    if (this->namespaces.contains("")) {
        ret = this->namespaces.at("")->search_impl(name);
    }
    auto names = Split(name);
    if (ret == nullptr) {
        if (names.size() >= 2) {
            auto ns = this->namespaces.find(names[0]);
            if (ns != this->namespaces.end()) {
                ret = ns->second->search_impl(name.substr(1));
            }
        } else if (names.size() == 1) {
            auto ns = this->values.find(name);
            if (ns != this->values.end()) {
                ret = ns->second;
            }
        }
    }
    return ret;
}

}  // namespace Analysis

std::vector<std::string_view> Split(const std::string_view str, const char delim) {
    std::vector<std::string_view> result;

    int indexCommaToLeftOfColumn = 0;
    int indexCommaToRightOfColumn = -1;

    for (int i = 0; i < static_cast<int>(str.size()); i++) {
        if (str[i] == delim) {
            indexCommaToLeftOfColumn = indexCommaToRightOfColumn;
            indexCommaToRightOfColumn = i;
            int index = indexCommaToLeftOfColumn + 1;
            int length = indexCommaToRightOfColumn - index;

            // Bounds checking can be omitted as logically, this code can never be invoked
            // Try it: put a breakpoint here and run the unit tests.
            /*if (index + length >= static_cast<int>(str.size()))
            {
                length--;
            }
            if (length < 0)
            {
                length = 0;
            }*/

            std::string_view column(str.data() + index, length);
            result.push_back(column);
        }
    }
    const std::string_view finalColumn(str.data() + indexCommaToRightOfColumn + 1,
                                       str.size() - indexCommaToRightOfColumn - 1);
    result.push_back(finalColumn);
    return result;
}