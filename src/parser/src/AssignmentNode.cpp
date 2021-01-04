#include "AssignmentNode.hpp"

llvm::Value* AST::AssignmentNode::codegen(Utils::IRContext* context) {
    // Generate LLVM value for the rval expression
    llvm::Value* rval = this->rVal->codeGen(context);
    // Store within the symbol table
    Utils::VarSymbolTable.at(Utils::VarSymbolTable.size() - 1)[this->name] = rval;
}