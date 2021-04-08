#include "UnaryExprNode.hpp"

#include <iostream>

#include "CodeGenUtils.hpp"
#include "TypeCheckingUtils.hpp"

/**
 * Code generation for unary elements.
 * We only consider each element as a single value and apply the operation as a function
 * This is based off the Binary operation code with the subtraction of the right hand side
 * of the expression being required, as LLVM handles this for us.
 * @param context
 * @return pointer to the resultant matrix initial position of the result
 */
llvm::Value* AST::UnaryExprNode::codeGen(Utils::IRContext* context) {
    // We go through and apply the relevant unary operator to each element of
    // the matrix
    llvm::Value* operand = this->operand->codeGen(context);

    auto operandMatNode = std::dynamic_pointer_cast<AST::ExprNode>(this->operand);
    llvm::Value* matAlloc = {};
    if (auto* operandType = std::get_if<Typing::MatrixType>(&*operandMatNode->type)) {
        matAlloc = Utils::createMatrix(context, *operandType);

        // We generate the operations sequentially
        // TODO: Add Kernel call for nvptx
        auto Builder = context->Builder;
        llvm::Function* parent = Builder->GetInsertBlock()->getParent();
        std::string opName = AST::UNA_OP_ENUM_STRING[this->op];

        llvm::BasicBlock* addBB = llvm::BasicBlock::Create(Builder->getContext(), opName + ".loop", parent);
        llvm::BasicBlock* endBB = llvm::BasicBlock::Create(Builder->getContext(), opName + ".done");

        auto indexAlloca = Utils::CreateEntryBlockAlloca(*Builder, "", llvm::Type::getInt64Ty(Builder->getContext()));
        auto* matSize = Utils::getLength(context, operand, *operandType);
        auto* nsize = Utils::getLength(context, matAlloc, *operandType);
        // parent->getBasicBlockList().push_back(addBB);
        Builder->CreateBr(addBB);

        Builder->SetInsertPoint(addBB);
        {
            auto* index = Builder->CreateLoad(indexAlloca);

            auto* lindex = Builder->CreateURem(index, matSize);
            auto* v = Utils::getValueFromMatrixPtr(context, operand, lindex, "lhs");

            llvm::Value* opResult;
            switch (op) {
                case NEG: {
                    opResult = context->Builder->CreateNeg(v, UNA_OP_ENUM_STRING[op]);
                    break;
                }
                case LNOT: {
                    opResult = context->Builder->CreateNot(v, UNA_OP_ENUM_STRING[op]);
                    break;
                }
                default: {
                    throw std::runtime_error("Unimplemented unary expression [" + std::string(UNA_OP_ENUM_STRING[op]) +
                                             "]");
                }
            }
            Utils::setValueFromMatrixPtr(context, matAlloc, index, opResult);

            // Update counter
            auto* next = Builder->CreateAdd(
                index, llvm::ConstantInt::get(context->module->getContext(), llvm::APInt{64, 1, true}), "add");
            Builder->CreateStore(next, indexAlloca);

            // Test if completed list
            auto* done = Builder->CreateICmpUGE(next, nsize);
            Builder->CreateCondBr(done, endBB, addBB);
        }

        parent->getBasicBlockList().push_back(endBB);
        Builder->SetInsertPoint(endBB);
    }
    return matAlloc;
}

void AST::UnaryExprNode::semanticPass(Utils::IRContext* context) {
    this->operand->semanticPass(context);
    Typing::MatrixType operandType = TypeCheckUtils::extractMatrixType(this->operand);
    Typing::PRIMITIVE primType = operandType.getPrimitiveType();
    switch (this->op) {
        case AST::UNA_OPERATORS::NEG:
            TypeCheckUtils::assertNumericType(primType);
            break;
        case AST::UNA_OPERATORS::BNOT:
            TypeCheckUtils::assertBooleanType(primType);
            break;
        case AST::UNA_OPERATORS::LNOT:
            TypeCheckUtils::assertLogicalType(primType);
    }
    this->type = std::make_shared<Typing::Type>(operandType);
}

/**
 * Function to calculate whenever or not we should execute the unary operation on the GPU
 * @param op
 * @return
 */
bool AST::UnaryExprNode::shouldExecuteGPU(Utils::IRContext * context, AST::UNA_OPERATORS op) {
    if(context->compilerOptions->optimisationLevel == OPTIMISATION::EXPERIMENTAL) {
        // Define a lookup table for the operation complexity
        auto operandMatrix = std::dynamic_pointer_cast<AST::ExprNode>(this->operand);
        auto* operandMatrixType = std::get_if<Typing::MatrixType>(&*operandMatrix->type);
        int entropy = operandMatrixType->getLength();
        int maxCPUEntropy = 400;  // 400 corresponds to 20x20 matrix
        return entropy >= maxCPUEntropy;
    }
    return true;
}


void AST::UnaryExprNode::dimensionPass(Analysis::DimensionSymbolTable* nt) {
    if (auto* mt = std::get_if<Typing::MatrixType>(&*type)) {
        operand->dimensionPass(nt);
        if (auto inner = std::get_if<Typing::MatrixType>(&*operand->type)) {
            mt->dimensions = inner->dimensions;
        }
    }
}
