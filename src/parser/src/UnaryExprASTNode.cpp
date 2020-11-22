#include "UnaryExprASTNode.hpp"

#include <iostream>

void AST::UnaryExprASTNode::codeGen(llvm::Module* module, llvm::Function* fp) {
    /* We need to work out what the type of the expr is!
     * By Default all raw values are some type of matrix, we need to determine
     * if we can get this type as a matrix or if it is something else
     */
    auto exprType = this->operand->type;
    if (exprType->isPrimitive) {
        // static cast to matrix
        auto* matType = static_cast<Typing::MatrixType*>(exprType.get());
        if (!matType) {
            std::cerr << "Internal Compiler Error: Matrix type for "
                      << exprType->name << " pointer is invalid!" << std::endl;
            return;
        }
        // We determine the size of offset for each of the values
        int offset = exprType->offset();
    }
}