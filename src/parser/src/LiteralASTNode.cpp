#include "LiteralASTNode.hpp"

template <class T>
void AST::LiteralASTNode<T>::codeGen(llvm::Module* module,
                                     llvm::IRBuilder<>* Builder,
                                     llvm::Function* fp) {}