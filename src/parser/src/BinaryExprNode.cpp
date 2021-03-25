#include "BinaryExprNode.hpp"

#include <CodeGenUtils.hpp>
#include <MatrixNode.hpp>
#include <TypeException.hpp>

#include "TypeCheckingUtils.hpp"

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

    auto* nValSize =
        Utils::getValueFromLLVM(context, static_cast<int>(lhsMat->dimensions.at(0)), Typing::PRIMITIVE::INT, false);
    auto* mValSize =
        Utils::getValueFromLLVM(context, static_cast<int>(rhsMat->dimensions.at(1)), Typing::PRIMITIVE::INT, false);
    auto* pValSize =
        Utils::getValueFromLLVM(context, static_cast<int>(rhsMat->dimensions.at(0)), Typing::PRIMITIVE::INT, false);

    // TODO: Propagate this to the upper thread loop
    auto* iValIndex = Utils::getValueFromLLVM(context, 0, Typing::PRIMITIVE::INT, false);
    auto* jValIndex = Utils::getValueFromLLVM(context, 0, Typing::PRIMITIVE::INT, false);

    // Initialise k to be 1
    auto* kValIndex = Utils::getValueFromLLVM(context, 1, Typing::PRIMITIVE::INT, false);

    /*
     * This is the inner most loop for the matrix multiplication algorithm
     * This will be executed in lockstep amongst the N*M processes spawned by CUDA
     * This should be called from the host and executed on the CUDA GPU as a function rather than
     * as a basic block. This provides us with a basic (not most efficient) CUDA Matrix Multiplication
     * algorithm.
     */
    llvm::BasicBlock* multFunctionLoopBB =
        llvm::BasicBlock::Create(context->module->getContext(), "matrixMult.loop", func);
    llvm::BasicBlock* multFunctionEndBB =
        llvm::BasicBlock::Create(context->module->getContext(), "matrixMult.end", func);
    context->Builder->CreateBr(multFunctionLoopBB);
    context->Builder->SetInsertPoint(multFunctionLoopBB);
    {
        // Loop invariant condition / conditional break if violated
        auto* cndr = context->Builder->CreateICmpNE(kValIndex, pValSize, "loopinv");
        context->Builder->CreateCondBr(cndr, multFunctionLoopBB, multFunctionEndBB);
        // Matrix multiplication for i,j at position k is A_(i,k) * B_(k,j)
        llvm::Value* a = Utils::getValueFromIndex(context, lhsVal, lhsMat, {iValIndex, kValIndex});
        llvm::Value* b = Utils::getValueFromIndex(context, lhsVal, lhsMat, {kValIndex, jValIndex});
        llvm::Value* tmpResult = context->Builder->CreateMul(a, b);
        // We sum from k = 1 to p, so add the result to itself
        result = context->Builder->CreateAdd(result, tmpResult);

        // Update the kIndex from the loop
        kValIndex =
            context->Builder->CreateAdd(kValIndex, Utils::getValueFromLLVM(context, 1, Typing::PRIMITIVE::INT, false));
        // Loop back up and check the loop invariant condition
        context->Builder->CreateBr(multFunctionLoopBB);
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
// op = MATM, CHAIN
void AST::BinaryExprNode::semanticPass(Utils::IRContext* context) {
    this->lhs->semanticPass(context);
    this->rhs->semanticPass(context);

    Typing::MatrixType lhsTy = TypeCheckUtils::extractMatrixType(this->lhs);
    Typing::MatrixType rhsTy = TypeCheckUtils::extractMatrixType(this->rhs);

    Typing::PRIMITIVE lhsPrim = lhsTy.getPrimitiveType();
    Typing::PRIMITIVE rhsPrim = rhsTy.getPrimitiveType();

    TypeCheckUtils::assertCompatibleTypes(lhsPrim, rhsPrim);
    Typing::PRIMITIVE primType;

    switch (this->op) {
        case AST::BIN_OPERATORS::BAND:
        case AST::BIN_OPERATORS::BOR:
            TypeCheckUtils::assertBooleanType(lhsPrim);
            TypeCheckUtils::assertBooleanType(rhsPrim);
            this->type = TypeCheckUtils::makeMatrixType(lhsTy.getDimensions(), lhsPrim);
            break;
        case AST::BIN_OPERATORS::PLUS:
        case AST::BIN_OPERATORS::MINUS:
        case AST::BIN_OPERATORS::MUL:
        case AST::BIN_OPERATORS::DIV:
        case AST::BIN_OPERATORS::POW:
            TypeCheckUtils::assertNumericType(lhsPrim);
            TypeCheckUtils::assertNumericType(rhsPrim);
            primType = TypeCheckUtils::getHighestType(lhsPrim, rhsPrim);
            this->type = TypeCheckUtils::makeMatrixType(lhsTy.getDimensions(), primType);
            break;
        case AST::BIN_OPERATORS::LOR:
        case AST::BIN_OPERATORS::LAND:
            TypeCheckUtils::assertLogicalType(lhsPrim);
            TypeCheckUtils::assertLogicalType(rhsPrim);
            primType = TypeCheckUtils::getHighestType(lhsPrim, rhsPrim);
            this->type = TypeCheckUtils::makeMatrixType(lhsTy.getDimensions(), primType);
            break;
        case AST::BIN_OPERATORS::LT:
        case AST::BIN_OPERATORS::GT:
        case AST::BIN_OPERATORS::LTE:
        case AST::BIN_OPERATORS::GTE:
            if ((not TypeCheckUtils::isBool(lhsPrim)) and (not TypeCheckUtils::isNone(lhsPrim))) {
                if ((not TypeCheckUtils::isBool(rhsPrim)) and (not TypeCheckUtils::isNone(rhsPrim))) {
                    this->type = TypeCheckUtils::makeMatrixType(lhsTy.getDimensions(), Typing::PRIMITIVE::BOOL);
                } else {
                    TypeCheckUtils::wrongTypeError("Expected: int, float, string", rhsPrim);
                }
            } else {
                TypeCheckUtils::wrongTypeError("Expected: int, float, string", lhsPrim);
            }
            break;
        case AST::BIN_OPERATORS::EQ:
        case AST::BIN_OPERATORS::NEQ:
            if (not TypeCheckUtils::isNone(lhsPrim) and not TypeCheckUtils::isNone(rhsPrim)) {
                this->type = TypeCheckUtils::makeMatrixType(lhsTy.getDimensions(), Typing::PRIMITIVE::BOOL);
            } else {
                if (TypeCheckUtils::isNone(lhsPrim)) {
                    TypeCheckUtils::wrongTypeError("Expected: int, float, string", lhsPrim);
                }
                TypeCheckUtils::wrongTypeError("Expected: int, float, string", rhsPrim);
            }
            break;
    }
}
