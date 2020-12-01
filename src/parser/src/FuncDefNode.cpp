#include "FuncDefNode.hpp"

llvm::Value* AST::FuncBodyExprNode::codeGen(llvm::Module* TheModule,
                                            llvm::IRBuilder<>* Builder,
                                            llvm::Function* fp) {
    // For this function, we need a new BasicBlock structure
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(
        TheModule->getContext(), "func" + funcSig->funcName, fp,
        Builder->GetInsertBlock());
    Builder->SetInsertPoint(bb);
    // We can begin to generate code for this function
    for (auto stmt : this->expr) {
        stmt->codeGen(TheModule, Builder, fp);
    }
}