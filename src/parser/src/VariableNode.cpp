#include "VariableNode.hpp"

llvm::Value* AST::VariableNode::codeGen(Utils::IRContext* context) {
    llvm::Value* storeVal =
        context->symbolTable->getValue(this->name, context->symbolTable->getCurrentFunction())->llvmVal;
    // If we have a slicing operation we apply it now
    if (this->variableSlicing) {
        storeVal = handleSlicing(context, storeVal);
    }
    return storeVal;
}

void AST::VariableNode::semanticPass() {
    if (this->variableSlicing) {
        this->variableSlicing->semanticPass();
    } else {
        throw std::runtime_error("[Internal Error] Variable slicing not generated correctly!");
    }
}
llvm::Value* AST::VariableNode::handleSlicing(Utils::IRContext* context, llvm::Value* val) {
    std::vector<std::pair<int, int>> currentIndicies;
    std::vector<std::pair<int, int>> prevIndicies;
    std::vector<std::variant<bool, std::vector<int>>> slicesVec = variableSlicing->slices;

    std::shared_ptr<Typing::MatrixType> matType = std::dynamic_pointer_cast<Typing::MatrixType>(this->type);
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

    // Take the maximal set of the matrix
    prevIndicies.emplace_back(0, matType->getLength() - 1);

    // We wish to generate a restricted set of indicies from the last dimension first
    auto rit = slices.rbegin();
    for (; rit != slices.rend(); ++rit) {
    }
    return nullptr;
}
