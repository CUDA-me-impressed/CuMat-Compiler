#include "BinaryExprNode.hpp"

#include <CodeGenUtils.hpp>
#include <MatrixNode.hpp>
#include <TypeException.hpp>

llvm::Value* AST::BinaryExprNode::codeGen(Utils::IRContext* context) {
    // Assumption is that our types are two evaluated matricies of compatible
    // dimensions. We first generate code for each of the l and r matricies
    llvm::Value* lhsVal = lhs->codeGen(context);
    llvm::Value* rhsVal = rhs->codeGen(context);
    auto lhsMatNode = std::dynamic_pointer_cast<AST::ExprNode>(this->lhs);
    auto rhsMatNode = std::dynamic_pointer_cast<AST::ExprNode>(this->rhs);

    if (auto* lhsType = std::get_if<Typing::MatrixType>(&*lhsMatNode->type)) {
        if (auto* rhsType = std::get_if<Typing::MatrixType>(&*rhsMatNode->type)) {
            auto lhsDimension = lhsType->getDimensions();
            auto rhsDimension = rhsType->getDimensions();

            Typing::MatrixType* resultType{};
            if (lhsDimension.size() > rhsDimension.size()) {
                resultType = lhsType;
            } else {
                resultType = rhsType;
            }

            auto newMatAlloc = Utils::createMatrix(context, *resultType);

            // TODO: Move to separate file and call from semantic pass function of AST nodes
            llvm::Type* intType = llvm::Type::getInt64Ty(context->module->getContext());
            llvm::Type* floatType = llvm::Type::getFloatTy(context->module->getContext());
            llvm::Type* lhsLLVMType = lhsType->getLLVMPrimitiveType(context);
            llvm::Type* rhsLLVMType = rhsType->getLLVMPrimitiveType(context);

            bool sameType = lhsLLVMType == rhsLLVMType;
            bool intOrFloat = (lhsLLVMType == intType or lhsLLVMType == floatType) and
                              (rhsLLVMType == intType or rhsLLVMType == floatType);

            switch (op) {
                case PLUS:
                case MINUS:
                case LOR: {
                    if (not(sameType and intOrFloat)) {
                        if (not sameType) {
                            Typing::mismatchTypeException("Types do not match");
                            std::exit(2);
                        } else {
                            if (not(lhsLLVMType == intType or lhsLLVMType == floatType)) {
                                Typing::wrongTypeException("Incorrect or unsupported type used", intType, lhsLLVMType);
                                std::exit(2);
                            } else {
                                Typing::wrongTypeException("Incorrect or unsupported type used", intType, rhsLLVMType);
                                std::exit(2);
                            }
                        }
                    }
                    elementWiseCodeGen(context, lhsVal, rhsVal, *lhsType, *rhsType, newMatAlloc, *resultType);
                    break;
                }
                default:
                    // TODO: Remove when ALL functions are implemented
                    break;
            }
        }
    }

    return nullptr;
}

void AST::BinaryExprNode::elementWiseCodeGen(Utils::IRContext* context, llvm::Value* lhsVal, llvm::Value* rhsVal,
                                             const Typing::MatrixType& lhsType, const Typing::MatrixType& rhsType,
                                             llvm::AllocaInst* matAlloc, const Typing::MatrixType& resType) {
    auto Builder = context->Builder;
    llvm::Function* parent = Builder->GetInsertBlock()->getParent();
    std::string opName = AST::BIN_OP_ENUM_STRING[this->op];

    llvm::BasicBlock* addBB = llvm::BasicBlock::Create(Builder->getContext(), opName + ".loop", parent);
    llvm::BasicBlock* endBB = llvm::BasicBlock::Create(Builder->getContext(), opName + ".done");

    auto indexAlloca = Utils::CreateEntryBlockAlloca(*Builder, "", llvm::Type::getInt64Ty(Builder->getContext()));
    auto* lsize = Utils::getLength(context, lhsVal, lhsType);
    auto* rsize = Utils::getLength(context, rhsVal, rhsType);
    auto* nsize = Utils::getLength(context, matAlloc, resType);
    // parent->getBasicBlockList().push_back(addBB);
    Builder->CreateBr(addBB);

    Builder->SetInsertPoint(addBB);
    {
        auto* index = Builder->CreateLoad(indexAlloca);

        auto* lindex = Builder->CreateURem(index, lsize);
        auto* rindex = Builder->CreateURem(index, rsize);
        auto* l = Utils::getValueFromMatrixPtr(context, lhsVal, lindex, "lhs");
        auto* r = Utils::getValueFromMatrixPtr(context, rhsVal, rindex, "rhs");
        auto* opResult = applyOperatorToOperands(context, this->op, l, r, opName);
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

/**
 * Abstraction out of LLVM CallInst to return the correct type for our binary tree.
 * @param op
 * @param lhs
 * @param rhs
 * @param name
 * @return
 */
llvm::Value* AST::BinaryExprNode::applyOperatorToOperands(Utils::IRContext* context, const AST::BIN_OPERATORS& op,
                                                          llvm::Value* lhs, llvm::Value* rhs, const std::string& name) {
    // TODO: Currently only works with integer values, will need to be extended to FP
    switch (op) {
        case PLUS: {
            return context->Builder->CreateAdd(lhs, rhs, name);
        }
        case MINUS: {
            return context->Builder->CreateSub(lhs, rhs, name);
        }
        case LOR: {
            return context->Builder->CreateOr(lhs, rhs, name);
        }
    }
}
