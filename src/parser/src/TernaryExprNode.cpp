#include "TernaryExprNode.hpp"

#include "CodeGenUtils.hpp"

llvm::Value *AST::TernaryExprNode::codeGen(Utils::IRContext *context) {
    // Generate the return value for the evaluate condition
    llvm::Value *conditionEval = this->condition->codeGen(context);
    if (!conditionEval) return nullptr;  // TODO: Handle errors gracefully

    auto matRecord = Utils::getMatrixFromPointer(context, conditionEval);
    if (matRecord.rank != Utils::getValueFromLLVM(context, 0, Typing::PRIMITIVE::INT, true)) return nullptr;
    // Fetch value from matrix memory
    llvm::Value *dataVal = Utils::getValueFromPointerOffset(context, matRecord.dataPtr, 0, "dataVal");

    if (!dataVal->getType()->isIntegerTy(1)) return nullptr;  // TODO: Handle errors gracefully
    conditionEval = context->Builder->CreateICmpNE(
            dataVal, llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(1, 0, true)), "ifcond");

    // Store function ptr for later use
    llvm::Function *func = context->Builder->GetInsertBlock()->getParent();

    llvm::BasicBlock *truthyBB = llvm::BasicBlock::Create(context->module->getContext(), "truthy", func);
    llvm::BasicBlock *mergeBB = llvm::BasicBlock::Create(context->module->getContext(), "merge");
    llvm::BasicBlock *falseyBB = llvm::BasicBlock::Create(context->module->getContext(), "falsey");

    // Handle the code of the body at the correct position
    context->Builder->SetInsertPoint(truthyBB);
    llvm::Value *truthyVal = this->truthy->codeGen(context);
    context->Builder->CreateBr(mergeBB);
    truthyBB = context->Builder->GetInsertBlock();

    func->getBasicBlockList().push_back(falseyBB);
    // Insert at the correct position
    context->Builder->SetInsertPoint(falseyBB);
    llvm::Value *falseyVal = this->falsey->codeGen(context);

    // Merge the lines of execution together
    context->Builder->CreateBr(mergeBB);
    func->getBasicBlockList().push_back(mergeBB);
    context->Builder->SetInsertPoint(mergeBB);

    return truthyVal;
}