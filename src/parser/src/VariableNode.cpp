#include "VariableNode.hpp"

llvm::Value* AST::VariableNode::codeGen(Utils::IRContext* context) {
    llvm::Value* storeVal = context->symbolTable->getValue(this->name);
    return storeVal;
}
