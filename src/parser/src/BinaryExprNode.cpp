#include "BinaryExprNode.hpp"

#include <CodeGenUtils.hpp>
#include <MatrixNode.hpp>

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

            switch (op) {
                case PLUS:
                case MINUS:
                case LOR: {
                    elementWiseCodeGen(context, lhsVal, rhsVal, *lhsType, *rhsType, newMatAlloc, *resultType);
                    break;
                }
                default:
                    throw std::runtime_error("Unimplemented binary expression [" + std::string(BIN_OP_ENUM_STRING[op]) +
                                             "]");
            }
        }
    }

    return nullptr;
}

void AST::BinaryExprNode::elementWiseCodeGen(Utils::IRContext* context, llvm::Value* lhsVal, llvm::Value* rhsVal,
                                             const Typing::MatrixType& lhsType, const Typing::MatrixType& rhsType,
                                             llvm::Instruction* matAlloc, const Typing::MatrixType& resType) {
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
        default: {
            throw std::runtime_error("Unimplemented binary expression [" + std::string(BIN_OP_ENUM_STRING[op]) + "]");
        }
    }
}
llvm::Value* AST::BinaryExprNode::matrixMultiply(Utils::IRContext* context, std::shared_ptr<Typing::MatrixType> lhsMat,
                                                 std::shared_ptr<Typing::MatrixType> rhsMat, llvm::Value* lhsVal,
                                                 llvm::Value* rhsVal) {
    /* PRE:
     * LHS is a matrix A_(i,p), RHS is a matrix B_(p,j)
     * We output a matrix C_(i,j)
     * */

    // We will generate a matrix multiplication function that CuMat programs will call directly
    // This should only be defined once per program
    if (lhsMat->rank != rhsMat->rank)
        throw std::runtime_error("Cannot compute matrix multiplication on matricies with different ranks!");
    // TODO: Sort out rank 1 multiplication (vector)
    if (!lhsMat->rank == 2 || !rhsMat->rank == 2)
        throw std::runtime_error("Matrix rank too high to compute matrix multiplication! Must be sliced first.");
    if (lhsMat->primType != rhsMat->primType) throw std::runtime_error("Matrix primitive types do not match");

    // Create the resultant matrix on device memory
    std::shared_ptr<Typing::MatrixType> resultType = std::make_shared<Typing::MatrixType>();
    resultType->primType = lhsMat->primType;
    resultType->rank = lhsMat->rank;
    resultType->dimensions = {rhsMat->getDimensions()[0], lhsMat->getDimensions()[1]};
    auto* mat = Utils::createMatrix(context, *resultType);

    // We create the multiplication function first, change some stuff and then revert into the current function
    // Returns a value we can insert into the matrix C at position i,j
    llvm::Type* i64Ty = llvm::Type::getInt64Ty(context->module->getContext());
    llvm::FunctionType* ft = llvm::FunctionType::get(
        resultType->getLLVMPrimitiveType(context),
        // Params: (lhsMat, rhsMat, lhsX, lhsY, rhsX, rhsY, i,j,p)
        {lhsVal->getType(), rhsVal->getType(), i64Ty, i64Ty, i64Ty, i64Ty, i64Ty, i64Ty, i64Ty}, false);
    llvm::Function* func = llvm::Function::Create(
        ft, llvm::Function::ExternalLinkage, "matMult_" + context->symbolTable->getCurrentFunction(), context->module);

    llvm::BasicBlock* multFunctionBB =
        llvm::BasicBlock::Create(context->module->getContext(), "matrixMult.entry", func);
    context->Builder->CreateBr(multFunctionBB);
    context->Builder->SetInsertPoint(multFunctionBB);

    // TODO: Floating point
    // Allocate return result for this entry
    llvm::Value* result = llvm::ConstantInt::get(resultType->getLLVMType(context), llvm::APInt(64, 0, true));

    auto* nValSize = Utils::getValueFromLLVM(context, static_cast<int>(lhsMat->dimensions.at(0)), Typing::PRIMITIVE::INT, false);
    auto* mValSize = Utils::getValueFromLLVM(context, static_cast<int>(rhsMat->dimensions.at(1)), Typing::PRIMITIVE::INT, false);
    auto* pValSize = Utils::getValueFromLLVM(context, static_cast<int>(rhsMat->dimensions.at(0)), Typing::PRIMITIVE::INT, false);

    // TODO: Propagate this to the upper thread loop
    auto* iValIndex = Utils::getValueFromLLVM(context, 0, Typing::PRIMITIVE::INT, false);
    auto* jValIndex = Utils::getValueFromLLVM(context, 0, Typing::PRIMITIVE::INT, false);

    // Initialise k to be 1
    auto* kValIndex = Utils::getValueFromLLVM(context, 1, Typing::PRIMITIVE::INT, false);

    llvm::BasicBlock* multFunctionLoopBB =
        llvm::BasicBlock::Create(context->module->getContext(), "matrixMult.loop", func);
    llvm::BasicBlock* multFunctionEndBB =
        llvm::BasicBlock::Create(context->module->getContext(), "matrixMult.end", func);
    context->Builder->CreateBr(multFunctionLoopBB);
    context->Builder->SetInsertPoint(multFunctionLoopBB);
    {
        // We create a loop that goes through all of the elements
        kValIndex = context->Builder->CreateAdd()
        auto* cndr = context->Builder->CreateICmpNE(kValIndex, pValSize, "loopinv");
        context->Builder->CreateCondBr(cndr, multFunctionLoopBB, multFunctionEndBB);
    }
    context->Builder->SetInsertPoint(multFunctionEndBB);


    // We wish to loop over each element in the matrix and spawn a CUDA thread for each one
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(
        context->module->getContext(), context->symbolTable->getCurrentFunction() + "_cudaSpawn", context->function);
    context->Builder->SetInsertPoint(bb);

    // Declare this new function to be a

    // We create a basic block and inherit this

    return nullptr;
}
