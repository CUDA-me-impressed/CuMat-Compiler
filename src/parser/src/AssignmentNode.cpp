#include "AssignmentNode.hpp"

#include <iostream>

llvm::Value* AST::AssignmentNode::codeGen(Utils::IRContext* context) {
    // Generate LLVM value for the rval expression
    llvm::Value* rVal = this->rVal->codeGen(context);

    if (!context->symbolTable->inSymbolTable(this->name, context->symbolTable->getCurrentFunction())) {
        // Something has gone wrong during the parse stage and we have not added the symbol into the table
        // Raising a warning!
        std::cout << "[Internal Warning] Symbol " << this->name
                  << " was not found within the symbol"
                     " table. Created during codegen"
                  << std::endl;
        // No typing information can be inferred at this stage (nullptr) - Can and will cause issues hence the warning
        context->symbolTable->setValue(nullptr, rVal, this->name, context->symbolTable->getCurrentFunction());
    } else {
        context->symbolTable->updateValue(rVal, this->name, context->symbolTable->getCurrentFunction());
    }
    return rVal;
}

void AST::AssignmentNode::semanticPass() {
    this->lVal->semanticPass();
    this->rVal->semanticPass();
}