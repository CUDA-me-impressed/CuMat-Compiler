#include "TernaryExprNode.hpp"

#include <DimensionPass.hpp>
#include <iostream>

#include "CodeGenUtils.hpp"
#include "TypeCheckingUtils.hpp"

llvm::Value* AST::TernaryExprNode::codeGen(Utils::IRContext* context) {
    // Generate the return value for the evaluate condition
    llvm::Value* conditionEval = this->condition->codeGen(context);

    if (!conditionEval) return nullptr;  // TODO: Handle errors gracefully

//    auto matRecord = Utils::getMatrixFromPointer(context, conditionEval);

    // Fetch value from matrix memory
//    llvm::Value* dataVal = Utils::getValueFromPointerOffset(context, matRecord.dataPtr, 0, "dataVal");

//    if (!dataVal->getType()->isIntegerTy(1)) {
//        llvm::Type* boolType = static_cast<llvm::Type*>(llvm::Type::getInt1Ty(context->module->getContext()));
//        std::exit(2);
//    }  // TODO: Handle errors gracefully
//    conditionEval = context->Builder->CreateICmpNE(
//        dataVal, llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(1, 0, true)), "ifcond");

    return truthy->codeGen(context);

//    // Store function ptr for later use
//    llvm::Function* func = context->Builder->GetInsertBlock()->getParent();
//
//    llvm::BasicBlock* truthyBB = llvm::BasicBlock::Create(context->module->getContext(), "truthy", func);
//    llvm::BasicBlock* mergeBB = llvm::BasicBlock::Create(context->module->getContext(), "merge");
//    llvm::BasicBlock* falseyBB = llvm::BasicBlock::Create(context->module->getContext(), "falsey");
//
//    llvm::Value* truthyVal = this->truthy->codeGen(context);
//    llvm::Value* falseyVal = this->falsey->codeGen(context);
//
//    auto* truthyType = std::get_if<Typing::MatrixType>(&*this->truthy->type);
//    if (this->truthy->isLiteralNode()) {
//        truthyVal = Utils::upcastLiteralToMatrix(context, *truthyType, truthyVal);
//    }
//
//    auto* falseyType = std::get_if<Typing::MatrixType>(&*this->falsey->type);
//    if (this->falsey->isLiteralNode()) {
//        falseyVal = Utils::upcastLiteralToMatrix(context, *falseyType, falseyVal);
//    }
//
//    llvm::Type* i32ty = llvm::Type::getInt32Ty(context->module->getContext());
//    auto dataAllocSize = llvm::ConstantExpr::getTruncOrBitCast(llvm::ConstantExpr::getSizeOf(truthyVal->getType()), i32ty);
//
//    auto* phi = Utils::createMatrix(context, *truthyType);
//    context->Builder->Insert(phi);
//
//    context->Builder->CreateCondBr(conditionEval, truthyBB, falseyBB);
//
//    // Handle the code of the body at the correct position
//    context->Builder->SetInsertPoint(truthyBB);
//    context->Builder->CreateMemCpy(phi, phi->getPointerAlignment(context->module->getDataLayout()), truthyVal,
//                                   truthyVal->getPointerAlignment(context->module->getDataLayout()),
//                                   llvm::ConstantExpr::getSizeOf(truthyVal->getType()));
//    context->Builder->CreateBr(mergeBB);
//
//    func->getBasicBlockList().push_back(falseyBB);
//    // Insert at the correct position
//    context->Builder->SetInsertPoint(falseyBB);
//    context->Builder->CreateMemCpy(phi, phi->getPointerAlignment(context->module->getDataLayout()), falseyVal,
//                                   falseyVal->getPointerAlignment(context->module->getDataLayout()),
//                                   llvm::ConstantExpr::getSizeOf(falseyVal->getType()));
//
//    // Merge the lines of execution together
//    context->Builder->CreateBr(mergeBB);
//
//    func->getBasicBlockList().push_back(mergeBB);
//    context->Builder->SetInsertPoint(mergeBB);
//
//    return phi;
}

void AST::TernaryExprNode::semanticPass(Utils::IRContext* context) {
    this->condition->semanticPass(context);
    this->truthy->semanticPass(context);
    this->falsey->semanticPass(context);
    Typing::MatrixType tTy = TypeCheckUtils::extractMatrixType(this->truthy);
    Typing::MatrixType fTy = TypeCheckUtils::extractMatrixType(this->falsey);
    TypeCheckUtils::assertMatchingTypes(tTy.getPrimitiveType(), fTy.getPrimitiveType());
    
    this->type = TypeCheckUtils::makeMatrixType(std::vector<uint>(), tTy.getPrimitiveType());
}
void AST::TernaryExprNode::dimensionPass(Analysis::DimensionSymbolTable* nt) {
    condition->dimensionPass(nt);
    truthy->dimensionPass(nt);
    falsey->dimensionPass(nt);
    auto* true_mt = std::get_if<Typing::MatrixType>(&*truthy->type);
    auto* false_mt = std::get_if<Typing::MatrixType>(&*falsey->type);
    auto* type = std::get_if<Typing::MatrixType>(this->type.get());
    if (true_mt && false_mt && type) {
        if (!expandableDimensionMatrix(*true_mt, *false_mt) || true_mt->rank != false_mt->rank) {
            dimension_error("If else block branches yield miss-matched dimension", this);
        }
        type->dimensions = expandedDimension(*true_mt, *false_mt);
        type->rank = type->dimensions.size();
    }
}
