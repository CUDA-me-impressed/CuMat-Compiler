#include "TernaryExprNode.hpp"

#include <DimensionPass.hpp>
#include <iostream>

#include "CodeGenUtils.hpp"
#include "TypeCheckingUtils.hpp"

llvm::Value* AST::TernaryExprNode::codeGen(Utils::IRContext* context) {
    // Generate the return value for the evaluate condition
    llvm::Value* conditionEval = this->condition->codeGen(context);

    if (!conditionEval) return nullptr;  // TODO: Handle errors gracefully

    auto matRecord = Utils::getMatrixFromPointer(context, conditionEval);
    if (matRecord.rank != Utils::getValueFromLLVM(context, 0, Typing::PRIMITIVE::INT, true)) return nullptr;
    // Fetch value from matrix memory
    llvm::Value* dataVal = Utils::getValueFromPointerOffset(context, matRecord.dataPtr, 0, "dataVal");

    if (!dataVal->getType()->isIntegerTy(1)) {
        llvm::Type* boolType = static_cast<llvm::Type*>(llvm::Type::getInt1Ty(context->module->getContext()));
        std::exit(2);
    }  // TODO: Handle errors gracefully
    conditionEval = context->Builder->CreateICmpNE(
        dataVal, llvm::ConstantInt::get(context->module->getContext(), llvm::APInt(1, 0, true)), "ifcond");

    // Store function ptr for later use
    llvm::Function* func = context->Builder->GetInsertBlock()->getParent();

    llvm::BasicBlock* truthyBB = llvm::BasicBlock::Create(context->module->getContext(), "truthy", func);
    llvm::BasicBlock* mergeBB = llvm::BasicBlock::Create(context->module->getContext(), "merge");
    llvm::BasicBlock* falseyBB = llvm::BasicBlock::Create(context->module->getContext(), "falsey");

    // Handle the code of the body at the correct position
    context->Builder->SetInsertPoint(truthyBB);
    llvm::Value* returnVal = this->truthy->codeGen(context);
    context->Builder->CreateBr(mergeBB);
    returnVal = context->Builder->GetInsertBlock();

    func->getBasicBlockList().push_back(falseyBB);
    // Insert at the correct position
    context->Builder->SetInsertPoint(falseyBB);
    returnVal = this->falsey->codeGen(context);

    // Merge the lines of execution together
    context->Builder->CreateBr(mergeBB);
    func->getBasicBlockList().push_back(mergeBB);
    context->Builder->SetInsertPoint(mergeBB);

    return returnVal;
}

void AST::TernaryExprNode::semanticPass(Utils::IRContext* context) {
    this->condition->semanticPass(context);
    this->truthy->semanticPass(context);
    this->falsey->semanticPass(context);
    Typing::MatrixType tTy = TypeCheckUtils::extractMatrixType(this->truthy);
    Typing::MatrixType fTy = TypeCheckUtils::extractMatrixType(this->falsey);
    if (TypeCheckUtils::isNone(tTy.getPrimitiveType()) or TypeCheckUtils::isNone(fTy.getPrimitiveType())) {
        if (TypeCheckUtils::isNone(tTy.getPrimitiveType()) and TypeCheckUtils::isNone(fTy.getPrimitiveType())) {
            std::cerr << "Both branches of a Ternary Expression cannot be recursive calls" << std::endl;
            std::exit(TypeCheckUtils::FUNCTION_ERROR);
        }
        if (TypeCheckUtils::isNone(tTy.getPrimitiveType())) {
            this->type = TypeCheckUtils::makeMatrixType(std::vector<uint>(), fTy.getPrimitiveType());
        } else {
            this->type = TypeCheckUtils::makeMatrixType(std::vector<uint>(), tTy.getPrimitiveType());
        }
    } else {
        TypeCheckUtils::assertMatchingTypes(tTy.getPrimitiveType(), fTy.getPrimitiveType());
        this->type = TypeCheckUtils::makeMatrixType(std::vector<uint>(), tTy.getPrimitiveType());
    }
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
