#include "BlockNode.hpp"

llvm::Value* AST::BlockNode::codeGen(llvm::Module* TheModule, llvm::IRBuilder<>* Builder, llvm::Function* fp) {
    return Node::codeGen(TheModule, Builder, fp);
}
