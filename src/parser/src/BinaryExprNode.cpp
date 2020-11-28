#include "BinaryExprNode.hpp"

llvm::Value* AST::BinaryExprNode::codeGen(llvm::Module* module,
                                          llvm::IRBuilder<>* Builder,
                                          llvm::Function* fp) {
    // Assumption is that our types are two evaluated matricies of compatible
    // dimensions. We first generate code for each of the l and r matricies
    //    Value* val =
    return nullptr;
}