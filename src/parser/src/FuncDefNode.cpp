#include "FuncDefNode.hpp"

llvm::Value* AST::FuncDefNode::codeGen(llvm::Module* TheModule,
                                            llvm::IRBuilder<>* Builder,
                                            llvm::Function* fp) {
    // For this function, we need a new BasicBlock structure
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(
        TheModule->getContext(), "func" + funcName, fp,
        Builder->GetInsertBlock());
    Builder->SetInsertPoint(bb);
    // TODO: Add Assignments here

    // We can begin to generate code for this function
    llvm::Value* returnVal = returnExpr->codeGen(TheModule, Builder, fp);
    // Return value should be a pointer to the first element of the matrix
    Builder->CreateRet(returnVal);
}