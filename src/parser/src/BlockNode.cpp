#include "BlockNode.hpp"

llvm::Value* AST::BlockNode::codeGen(Utils::IRContext* context) {
    // For this function, we need a new BasicBlock structure
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(context->module->getContext(), this->callingFunctionName + "_entry",
                                                    context->function);
    context->Builder->SetInsertPoint(bb);

    // Loop over each assignment in order
    for (auto ass : this->assignments) {
        ass->codegen(context);
    }

    // Generate Return statement code
    llvm::Value* returnExprVal = this->returnExpr->codeGen(context);
    llvm::Value* retVal = context->Builder->CreateRet(returnExprVal);

    return retVal;
}
