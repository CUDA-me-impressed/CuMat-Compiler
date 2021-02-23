#include "VariableNode.hpp"
#include <valarray>

llvm::Value* AST::VariableNode::codeGen(Utils::IRContext* context) {
    llvm::Value* storeVal =
        context->symbolTable->getValue(this->name, context->symbolTable->getCurrentFunction())->llvmVal;
    // If we have a slicing operation we apply it now
    if (this->variableSlicing) {
        storeVal = handleSlicing(context, storeVal);
    }
    return storeVal;
}

void AST::VariableNode::semanticPass(Utils::IRContext* context) {
    if (this->variableSlicing) {
        this->variableSlicing->semanticPass(context);
    } else {
        throw std::runtime_error("[Internal Error] Variable slicing not generated correctly!");
    }
}
llvm::Value* AST::VariableNode::handleSlicing(Utils::IRContext* context, llvm::Value* val) {
    std::vector<std::pair<int, int>> currentIndicies;
    std::vector<std::pair<int, int>> prevIndicies;
    std::vector<std::variant<bool, std::vector<int>>> slicesVec = variableSlicing->slices;

    Typing::MatrixType* matType = std::get_if<Typing::MatrixType>(&*this->type);
    if (!matType) {
        throw std::runtime_error("[Internal Error] Attempt to slice non-matrix type propagated to codegen");
    }

    std::vector<std::pair<int, int>> slices;
    // Give me a full vector of pairs :)
    for (int i = 0; i < matType->dimensions.size(); i++) {
        if (slicesVec.size() <= i || std::get_if<bool>(&slicesVec.at(i))) {
            slices.emplace_back(0, matType->dimensions.at(i));
        } else {
            auto sliceElement = *std::get_if<std::vector<int>>(&slicesVec.at(i));
            if (sliceElement.size() == 2) {
                slices.emplace_back(sliceElement.at(0), sliceElement.emplace_back(1));
            } else {
                slices.emplace_back(sliceElement.at(0), matType->dimensions.at(i));
            }
        }
    }

    // Lloyd's algorithm to calculate indicies from slices
    std::vector<int> firstSlices;
    std::transform(slices.begin(), slices.end(), std::back_inserter(firstSlices),
                   [] (std::pair<int,int> slice) -> int { return slice.first; });
    std::vector<std::vector<int>> groupIndicies;
    groupIndicies.emplace_back(firstSlices);
    int groupLength = slices[0].second - slices[0].first + 1;
    for(int i = 1; i < slices.size(); i++){
        int number = slices[i].second - slices[i].first +1;
        std::vector<std::vector<int>> lastGroupIndicies;
        std::copy(groupIndicies.begin(), groupIndicies.end(), std::back_inserter(lastGroupIndicies));
        groupIndicies = {};
        for(int x = 0; x < number; x++){
            for(auto gi : groupIndicies){
                int j = 0;
                std::transform(gi.begin(), gi.end(), gi.begin(), [&](int c) -> int {j++; return (j==i) ? c + x : c;});
            }
        }
    }
    return nullptr;
}
