#include "InputFileNode.hpp"

#include <iostream>
#include <map>
#include <numeric>
#include <vector>

#include "CodeGenUtils.hpp"
#include "DimensionsSymbolTable.hpp"
#include "TreePrint.hpp"
#include "TypeCheckingUtils.hpp"
#include "VariableNode.hpp"

llvm::Value* AST::InputFileNode::codeGen(Utils::IRContext* context) {
    //TODO: CALL C++ FUNCTION FOR THIS

    return nullptr;

}

void AST::InputFileNode::semanticPass(Utils::IRContext* context) {
    //TODO: Very little
}

std::string AST::InputFileNode::toTree(const std::string& prefix, const std::string& childPrefix) const {
    //TODO: Optional
}


void AST::InputFileNode::dimensionPass(Analysis::DimensionSymbolTable* nt) {
    //TODO: Just acknowledge?
}
