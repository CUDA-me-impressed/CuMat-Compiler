#include "LiteralASTNode.hpp"

template <class T>
llvm::Value* AST::LiteralASTNode<T>::codeGen(llvm::Module* module,
                                             llvm::IRBuilder<>* Builder,
                                             llvm::Function* fp) {
    return nullptr;
}