#include "VariableNode.hpp"

llvm::Value* AST::VariableNode::codegen(Utils::IRContext* context){
    // Generate LLVM value for the rval expression
    llvm::Value* rval = this->rval->codeGen(context);
    // Store within the symbol table
    Utils::AllocSymbolTable.at(Utils::AllocSymbolTable.size()-1)[this->name] = rval;

}