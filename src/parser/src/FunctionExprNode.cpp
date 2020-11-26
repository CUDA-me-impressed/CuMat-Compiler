#include "FunctionExprNode.hpp"

llvm::Value* AST::FunctionExprNode::codeGen(llvm::Module* module,
                                               llvm::IRBuilder<>* Builder,
                                               llvm::Function* fp) {
    return nullptr;
}