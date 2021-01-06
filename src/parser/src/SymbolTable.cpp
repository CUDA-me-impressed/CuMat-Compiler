#include "SymbolTable.hpp"

/**
 * Returns the LLVM Value stored in relation to the currently in scope value
 * @param symbolName
 * @return
 */
llvm::Value* Utils::SymbolTable::getValue(const std::string& symbolName) {
    if (inScope(symbolName)) {
        // We get the max value from the hash table
        auto locations = this->variableLocations[symbolName];
        // Get the max value where the element is stored within the locations and return the depth
        int curScope =
            locations.at(std::distance(locations.begin(), std::max_element(locations.begin(), locations.end())));
        return this->data.at(curScope)[symbolName];
    } else {
        // TODO: Graceful error handling -> Variable accessed outside of symbol table scope. Illegal access probs
        return nullptr;
    }
}

/**
 *
 * @param symbolName
 * @param storeVal
 * @return If value overwritten or not
 */
void Utils::SymbolTable::setValue(const std::string& symbolName, llvm::Value* storeVal) {
    // We need to update the locations
    if (!this->variableLocations.contains(symbolName)) {
        // Create a new vector
        this->variableLocations.insert(std::pair<std::string, std::vector<int>>(symbolName, std::vector<int>()));
    }

    // If empty or not the correct scope, we should add the scope index
    if (this->variableLocations[symbolName].empty() ||
        this->variableLocations[symbolName].back() != this->data.size()) {
        this->variableLocations[symbolName].emplace_back(data.size() - 1);
    }
    // Store the value in the map for this scope
    data.back()[symbolName] = storeVal;
}

bool Utils::SymbolTable::inScope(const std::string& symbolName) {
    // Just check if the location is stored for us
    return this->variableLocations.contains(symbolName);
}

void Utils::SymbolTable::newScope() {
    // We create a new element and place it at the back of the scope
    this->data.emplace_back();
}
void Utils::SymbolTable::exitScope() {}
