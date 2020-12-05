#include "BlockNode.hpp"

llvm::Value* AST::BlockNode::codeGen(Utils::IRContext* context) {
    // For this function, we need a new BasicBlock structure
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(context->module->getContext(), "func" + this->literalText,
                                                    context->function, context->Builder->GetInsertBlock());
    context->Builder->SetInsertPoint(bb);

    // TODO: Deal with the assignments

    // Generate Return statement code
    llvm::Value* returnExprVal = this->returnExpr->codeGen(context);
    llvm::Value* retVal = context->Builder->CreateRet(returnExprVal);
    return retVal;
}
