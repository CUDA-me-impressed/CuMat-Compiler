#include "FuncDefNode.hpp"

llvm::Value* AST::FuncDefNode::codeGen(Utils::IRContext* context) {
    // For this function, we need a new BasicBlock structure
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(
        context->module->getContext(), "func" + funcName, context->Builder->GetInsertBlock()->getParent(),
        context->Builder->GetInsertBlock());
    context->Builder->SetInsertPoint(bb);
    // TODO: Add Assignments here

    // We can begin to generate code for this function
    llvm::Value* returnVal = returnExpr->codeGen(context);
    // Return value should be a pointer to the first element of the matrix
    context->Builder->CreateRet(returnVal);
    return returnVal;
}