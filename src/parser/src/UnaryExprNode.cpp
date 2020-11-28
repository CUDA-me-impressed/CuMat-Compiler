#include "UnaryExprNode.hpp"

#include <iostream>

llvm::Value* AST::UnaryExprNode::codeGen(llvm::Module* module,
                                         llvm::IRBuilder<>* Builder,
                                         llvm::Function* fp) {
    // opval should be an evaluated matrix for which we can create a new matrix
    llvm::Value* opVal = this->operand->codeGen(module, Builder, fp);
    switch(this->op){
        case UNA_OPERATORS::NEG: {

        }
    }
    return nullptr;
}