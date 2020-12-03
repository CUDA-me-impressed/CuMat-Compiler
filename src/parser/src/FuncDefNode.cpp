#include "FuncDefNode.hpp"

llvm::Value* AST::FuncDefNode::codeGen(Utils::IRContext* context) {
    // For this function, we need a new BasicBlock structure
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(context->module->getContext(), "func" + this->funcName, context->Builder->GetInsertBlock()->getParent(), context->Builder->GetInsertBlock());
    context->Builder->SetInsertPoint(bb);
    // TODO: Deal with the assignments

    // Generate Return statement code
    llvm::Value* retVal = returnExpr->codeGen(context);
    context->Builder->CreateRet(retVal);
}