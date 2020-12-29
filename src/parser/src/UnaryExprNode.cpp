#include "UnaryExprNode.hpp"

#include "CodeGenUtils.hpp"

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
    llvm::Value* matAlloc;
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