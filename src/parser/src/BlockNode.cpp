#include "BlockNode.hpp"

llvm::Value* AST::BlockNode::codeGen(Utils::IRContext* context) {
    // For this function, we need a new BasicBlock structure
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(context->module->getContext(), this->callingFunctionName + "_entry",
                                                    context->function);
    context->Builder->SetInsertPoint(bb);

    // Loop over each assignment in order
    for (const auto ass : this->assignments) {
        ass->codeGen(context);
    }

    // Generate Return statement code
    llvm::Value* returnExprVal = this->returnExpr->codeGen(context);
    llvm::Value* retVal = context->Builder->CreateRet(returnExprVal);

    return retVal;
}

void AST::BlockNode::semanticPass(Utils::IRContext* context) {
    // Run semantic pass on all related nodes - block is not an Expr node, so does not have a type to assign
    for (auto const& assignment : this->assignments) assignment->semanticPass(context);
    this->returnExpr->semanticPass(context);
}