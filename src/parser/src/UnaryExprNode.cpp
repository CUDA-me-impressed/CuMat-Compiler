#include "UnaryExprNode.hpp"

#include "CodeGenUtils.hpp"
#include "MatrixNode.hpp"

llvm::Value* AST::UnaryExprNode::codeGen(llvm::Module* module,
                                         llvm::IRBuilder<>* Builder,
                                         llvm::Function* fp) {
    // opval should be an evaluated matrix for which we can create a new matrix
    llvm::Value* opVal = this->operand->codeGen(module, Builder, fp);
    // We go through and apply the relevant unary operator to each element of
    // the matrix
    auto matType = std::dynamic_pointer_cast<AST::MatrixNode>(this->operand);
    llvm::Type* ty = matType->getLLVMType(module);
    auto dimension = matType->getDimensions();
    auto newMatAlloc = Utils::generateMatrixAllocation(ty, dimension, Builder);
    // We generate the operations sequentially
    // TODO: Add Kernel call for nvptx

    switch (this->op) {
        case NEG: {
            break;
        }
        case LNOT:
            break;
        case BNOT:
            break;
    }
    return nullptr;
}