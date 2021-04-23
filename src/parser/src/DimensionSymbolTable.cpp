#include <string_view>

#include "DimensionsSymbolTable.hpp"
std::vector<std::string_view> $tb_split(std::string_view str, char delim = '.');

namespace Analysis {

std::shared_ptr<Typing::Type> DimensionSymbolTable::search_impl(const std::string_view name) noexcept {
    std::shared_ptr<Typing::Type> ret{};
    // try searching root namespace if it exists

    auto names = $tb_split(name);
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
    if (ret == nullptr) {
        if (this->namespaces.contains("")) {
            ret = this->namespaces.at("")->search_impl(name);
        }
    }
    return std::move(ret);
}

void DimensionSymbolTable::add_node(std::string name, std::shared_ptr<Typing::Type> type) noexcept {
    this->values.emplace(std::move(name), std::move(type));
}

}  // namespace Analysis

// cribbed from stackoverflow
// retrieved from https://stackoverflow.com/a/58048821 fetched 2021-04-10
std::vector<std::string_view> $tb_split(const std::string_view str, const char delim) {
    std::vector<std::string_view> result;

    int indexCommaToLeftOfColumn = 0;
    int indexCommaToRightOfColumn = -1;

    for (int i = 0; i < static_cast<int>(str.size()); i++) {
        if (str[i] == delim) {
            indexCommaToLeftOfColumn = indexCommaToRightOfColumn;
            indexCommaToRightOfColumn = i;
            int index = indexCommaToLeftOfColumn + 1;
            int length = indexCommaToRightOfColumn - index;

            std::string_view column(str.data() + index, length);
            result.push_back(column);
        }
    }
    const std::string_view finalColumn(str.data() + indexCommaToRightOfColumn + 1,
                                       str.size() - indexCommaToRightOfColumn - 1);
    result.push_back(finalColumn);
    return result;
}