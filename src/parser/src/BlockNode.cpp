#include "BlockNode.hpp"

llvm::Value* AST::BlockNode::codeGen(Utils::IRContext* context) {
    // We need to handle scope within the block
    context->symbolTable->newScope();

    // For this function, we need a new BasicBlock structure
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(context->module->getContext(), "func" + this->callingFunctionName,
                                                    context->function, context->Builder->GetInsertBlock());
    context->Builder->SetInsertPoint(bb);

    // Loop over each assignment in order
    for (auto ass : this->assignments) {
        ass->codegen(context);
    }

    // Generate Return statement code
    llvm::Value* returnExprVal = this->returnExpr->codeGen(context);
    llvm::Value* retVal = context->Builder->CreateRet(returnExprVal);

    context->symbolTable->exitScope();

    return retVal;
}
