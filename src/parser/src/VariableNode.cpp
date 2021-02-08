#include "VariableNode.hpp"

llvm::Value* AST::VariableNode::codeGen(Utils::IRContext* context) {
    llvm::Value* storeVal =
        context->symbolTable->getValue(this->name, context->symbolTable->getCurrentFunction())->llvmVal;
    return storeVal;
}

void AST::VariableNode::semanticPass() {
    this->variableSlicing->semanticPass();
}