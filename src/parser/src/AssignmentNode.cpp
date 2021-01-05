#include "AssignmentNode.hpp"

llvm::Value* AST::AssignmentNode::codegen(Utils::IRContext* context) {
    // Generate LLVM value for the rval expression
    llvm::Value* rVal = this->rVal->codeGen(context);
    // Store within the symbol table
    context->symbolTable->setValue(this->name, rVal);
    return rVal;
}