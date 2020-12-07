#include "BinaryExprNode.hpp"

#include <CodeGenUtils.hpp>
#include <MatrixNode.hpp>
#include <TypeException.hpp>

static llvm::AllocaInst* CreateEntryBlockAlloca(llvm::IRBuilder<>& Builder, const std::string& VarName,
                                                llvm::Type* Type) {
    llvm::IRBuilder<> TmpB(&Builder.GetInsertBlock()->getParent()->getEntryBlock(),
                           Builder.GetInsertBlock()->getParent()->getEntryBlock().begin());
    return TmpB.CreateAlloca(Type, nullptr, VarName);
}

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
            llvm::Type* lhsLLVMType = lhsType->getLLVMType(context);
            llvm::Type* rhsLLVMType = rhsType->getLLVMType(context);

            bool sameType = lhsLLVMType == rhsLLVMType;
            bool intOrFloat = (lhsLLVMType == intType or lhsLLVMType == floatType) and
                              (rhsLLVMType == intType or rhsLLVMType == floatType);

            switch (op) {
                case PLUS: {
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
                    plusCodeGen(context, lhsVal, rhsVal, *lhsType, *rhsType, newMatAlloc, *resultType);
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

void AST::BinaryExprNode::plusCodeGen(Utils::IRContext* context, llvm::Value* lhsVal, llvm::Value* rhsVal,
                                      const Typing::MatrixType& lhsType, const Typing::MatrixType& rhsType,
                                      llvm::AllocaInst* matAlloc, const Typing::MatrixType& resType) {
    auto Builder = context->Builder;
    llvm::Function* parent = Builder->GetInsertBlock()->getParent();

    llvm::BasicBlock* addBB = llvm::BasicBlock::Create(Builder->getContext(), "add.loop", parent);
    llvm::BasicBlock* endBB = llvm::BasicBlock::Create(Builder->getContext(), "add.done", parent);

    auto indexAlloca = CreateEntryBlockAlloca(*Builder, "", llvm::Type::getInt64Ty(Builder->getContext()));
    auto* lsize = Utils::getLength(context, lhsVal, lhsType);
    auto* rsize = Utils::getLength(context, rhsVal, rhsType);
    auto* nsize = Utils::getLength(context, matAlloc, resType);
    Builder->CreateBr(addBB);

    Builder->SetInsertPoint(addBB);
    {
        auto* index = Builder->CreateLoad(indexAlloca, "add.loadcounter");

        auto* lindex = Builder->CreateURem(index, lsize);
        auto* rindex = Builder->CreateURem(index, rsize);
        auto* l = Utils::getValueRelativeToPointer(context, lhsType.getLLVMType(context), lhsVal, lindex);
        auto* r = Utils::getValueRelativeToPointer(context, rhsType.getLLVMType(context), rhsVal, rindex);
        auto* add = Builder->CreateAdd(l, r, "add");
        Utils::insertRelativeToPointer(context, resType.getLLVMType(context), matAlloc, index, add);

        // Update counter
        auto* next = Builder->CreateAdd(
            index, llvm::ConstantInt::get(context->module->getContext(), llvm::APInt{64, 1, true}), "add");
        Builder->CreateStore(next, indexAlloca, "add.storecounter");

        // Test if completed list
        auto* done = Builder->CreateICmpUGE(next, nsize);
        Builder->CreateCondBr(done, endBB, addBB);
    }

    parent->getBasicBlockList().push_back(endBB);
    Builder->SetInsertPoint(endBB);
}
