#include "BinaryExprNode.hpp"

#include <CodeGenUtils.hpp>
#include <MatrixNode.hpp>

#include "CompilerOptions.hpp"
#include "TreePrint.hpp"
#include "TypeCheckingUtils.hpp"

namespace AST {

llvm::Value* AST::BinaryExprNode::codeGen(Utils::IRContext* context) {
    // We need to determine whenever or not we apply the CPU code, ultimately this is determined by the complexity of
    // the operation (i.e. if it would be simpler to just execute on the CPU, and complexity of the operation for us
    // i.e. not MAT-MAT mult)
    // Assumption is that our types are two evaluated matricies of compatible
    // dimensions. We first generate code for each of the l and r matricies
    llvm::Value* lhsVal = lhs->codeGen(context);
    llvm::Value* rhsVal = rhs->codeGen(context);
    auto lhsMatNode = std::dynamic_pointer_cast<AST::ExprNode>(this->lhs);
    auto rhsMatNode = std::dynamic_pointer_cast<AST::ExprNode>(this->rhs);
    llvm::Value* newMatAlloc;

    if (auto* lhsType = std::get_if<Typing::MatrixType>(&*lhsMatNode->type)) {
        if (auto* rhsType = std::get_if<Typing::MatrixType>(&*rhsMatNode->type)) {

            // Upcasting literal to matrix type
            if(lhsType->rank == 0){
                lhsVal = Utils::upcastLiteralToMatrix(context, *lhsType, lhsVal);
            }
            if(rhsType->rank == 0){
                rhsVal = Utils::upcastLiteralToMatrix(context, *rhsType, rhsVal);
            }

            auto lhsDimension = lhsType->getDimensions();
            auto rhsDimension = rhsType->getDimensions();

            auto* resType = std::get_if<Typing::MatrixType>(&*type);
            if (!resType) {
                throw std::runtime_error("Resultant Matrix Type not determined!");
            }

            newMatAlloc = Utils::createMatrix(context, *resType);

            if (true || shouldExecuteGPU(context, op) || this->op == BIN_OPERATORS::MATM) {
                auto lhsRecord = Utils::getMatrixFromPointer(context, lhsVal);
                auto rhsRecord = Utils::getMatrixFromPointer(context, rhsVal);
                auto resRecord = Utils::getMatrixFromPointer(context, newMatAlloc);
                llvm::Type* dataPtrType = llvm::Type::getInt64PtrTy(context->module->getContext());

                if (this->op != BIN_OPERATORS::MATM) {
                    llvm::Value* resLenLLVM = llvm::ConstantInt::get(
                        llvm::Type::getInt64Ty(context->module->getContext()), resType->getLength());
                    std::vector<llvm::Value*> argVals(
                        {lhsRecord.dataPtr, rhsRecord.dataPtr, resRecord.dataPtr, resLenLLVM});

                    if (lhsType->primType == Typing::PRIMITIVE::INT && rhsType->primType == Typing::PRIMITIVE::INT) {
                        auto callRet = context->Builder->CreateCall(
                            context->symbolTable->binaryFunctions[this->op].funcInt, argVals);
                    } else if (lhsType->primType == Typing::PRIMITIVE::FLOAT &&
                               rhsType->primType == Typing::PRIMITIVE::FLOAT) {
                        auto callRet = context->Builder->CreateCall(
                            context->symbolTable->binaryFunctions[this->op].funcFloat, argVals);
                    }
                } else {
                    auto* lenType = llvm::Type::getInt64Ty(context->module->getContext());
                    llvm::Value* lenI = llvm::ConstantInt::get(lenType, lhsType->dimensions[0]);
                    llvm::Value* lenK = llvm::ConstantInt::get(lenType, lhsType->dimensions[1]);
                    llvm::Value* lenJ = llvm::ConstantInt::get(lenType, lhsType->dimensions[1]);

                    std::vector<llvm::Value*> argVals(
                        {lhsRecord.dataPtr, rhsRecord.dataPtr, resRecord.dataPtr, lenI, lenK, lenJ});

                    if (lhsType->primType == Typing::PRIMITIVE::INT && rhsType->primType == Typing::PRIMITIVE::INT) {
                        auto callRet = context->Builder->CreateCall(
                            context->symbolTable->binaryFunctions[this->op].funcInt, argVals);
                    } else if (lhsType->primType == Typing::PRIMITIVE::FLOAT &&
                               rhsType->primType == Typing::PRIMITIVE::FLOAT) {
                        auto callRet = context->Builder->CreateCall(
                            context->symbolTable->binaryFunctions[this->op].funcFloat, argVals);
                    }
                }

            } else {
                // Execute this operation on CPU

                if (op != BIN_OPERATORS::MATM) {
                    elementWiseCodeGen(context, lhsVal, rhsVal, *lhsType, *rhsType, (llvm::Instruction*)newMatAlloc,
                                       *resType);
                } else {
                    throw std::runtime_error("Unimplemented binary expression [" + std::string(BIN_OP_ENUM_STRING[op]) +
                                             "]");
                }
            }
        }
    }

    return newMatAlloc;
}

llvm::Value* AST::BinaryExprNode::elementWiseCodeGen(Utils::IRContext* context, llvm::Value* lhsVal,
                                                     llvm::Value* rhsVal, const Typing::MatrixType& lhsType,
                                                     const Typing::MatrixType& rhsType, llvm::Instruction* matAlloc,
                                                     const Typing::MatrixType& resType) {
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

    return matAlloc;
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
    // TODO: Upcast to float -> Loss of precision? Discuss.

    if (lhs->getType()->isIntegerTy(64) && rhs->getType()->isIntegerTy(64)) {
        // Handle the integer-integer operations
        switch (op) {
            case PLUS: {
                return context->Builder->CreateAdd(lhs, rhs, name);
            }
            case MINUS: {
                return context->Builder->CreateSub(lhs, rhs, name);
            }
            case MUL: {
                return context->Builder->CreateMul(lhs, rhs, name);
            }
            case DIV: {
                return context->Builder->CreateSDiv(lhs, rhs, name);
            }
            case LOR: {
                // We use CreateBinOp because it refers to logical version rather than bitwise
                // See https://github.com/llvm-mirror/llvm/blob/master/include/llvm/IR/Instruction.def for the operators
                return context->Builder->CreateBinOp(llvm::Instruction::BinaryOps::Or, lhs, rhs, name);
            }
            case LAND: {
                return context->Builder->CreateBinOp(llvm::Instruction::BinaryOps::And, lhs, rhs, name);
            }
            // Dealing with comparison operators
            case EQ: {
                return context->Builder->CreateICmpEQ(lhs, rhs, name);
            }
            case NEQ: {
                return context->Builder->CreateICmpNE(lhs, rhs, name);
            }
            case LT: {
                return context->Builder->CreateICmpSLT(lhs, rhs, name);
            }
            case GT: {
                return context->Builder->CreateICmpSGT(lhs, rhs, name);
            }
            case GTE: {
                return context->Builder->CreateICmpSGE(lhs, rhs, name);
            }
            case LTE: {
                return context->Builder->CreateICmpSLE(lhs, rhs, name);
            }
            case BAND: {
                return context->Builder->CreateAnd(lhs, rhs, name);
            }
            case BOR: {
                return context->Builder->CreateOr(lhs, rhs, name);
            }
            case POW: {
                return applyPowerToOperands(context, lhs, rhs, false, name);
            }

            default: {
                if (context->compilerOptions->warningVerbosity == WARNINGS::INFO ||
                    context->compilerOptions->warningVerbosity == WARNINGS::ALL) {
                    throw std::runtime_error("Unimplemented binary expression [" + std::string(BIN_OP_ENUM_STRING[op]) +
                                             "]");
                }
            }
        }
    } else if (lhs->getType()->isDoubleTy() && rhs->getType()->isDoubleTy()) {
        // Handle the float-float operations
        switch (op) {
            case PLUS: {
                return context->Builder->CreateFAdd(lhs, rhs, name);
            }
            case MINUS: {
                return context->Builder->CreateFSub(lhs, rhs, name);
            }
            case MUL: {
                return context->Builder->CreateFMul(lhs, rhs, name);
            }
            case DIV: {
                return context->Builder->CreateFDiv(lhs, rhs, name);
            }
        }
    } else if (lhs->getType()->isIntegerTy(1) && rhs->getType()->isIntegerTy(1)) {
        // Handle boolean functions
        switch (op) {
            case PLUS:
            case MINUS: {
                return context->Builder->CreateXor(lhs, rhs, name);  // Add / Sub on GF(2) is equivalent to XOR
            }
            case MUL: {
                return context->Builder->CreateAnd(lhs, rhs, name);  // Multiplication on GF(2) is AND
            }
            case DIV: {
                break;  // Division on booleans not defined
            }
        }
    }

    if (context->compilerOptions->warningVerbosity == WARNINGS::INFO ||
        context->compilerOptions->warningVerbosity == WARNINGS::ALL) {
        // TODO: Better reporting
        throw std::runtime_error("Mutli-dimensional array multiplication occurred between two undefined types!");
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
    if (lhsMat->rank != 2 || rhsMat->rank != 2)
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
// Still to do - op = CHAIN , Needs function types sorted
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
            if ((not TypeCheckUtils::isString(lhsPrim)) and (not TypeCheckUtils::isNone(lhsPrim))) {
                if ((not TypeCheckUtils::isString(rhsPrim)) and (not TypeCheckUtils::isNone(rhsPrim))) {
                    if (TypeCheckUtils::isBool(lhsPrim)) {
                        TypeCheckUtils::assertMatchingTypes(lhsPrim, rhsPrim);
                        this->type = TypeCheckUtils::makeMatrixType(lhsTy.getDimensions(), Typing::PRIMITIVE::BOOL);
                    } else {
                        TypeCheckUtils::assertNumericType(lhsPrim);
                        TypeCheckUtils::assertNumericType(rhsPrim);
                        primType = TypeCheckUtils::getHighestType(lhsPrim, rhsPrim);
                        this->type = TypeCheckUtils::makeMatrixType(lhsTy.getDimensions(), primType);
                        break;
                    }
                } else {
                    TypeCheckUtils::wrongTypeError("Expected: int, float, bool", rhsPrim);
                }
            } else {
                TypeCheckUtils::wrongTypeError("Expected: int, float, bool", lhsPrim);
            }
            break;
        case AST::BIN_OPERATORS::MINUS:
        case AST::BIN_OPERATORS::MUL:
        case AST::BIN_OPERATORS::DIV:
        case AST::BIN_OPERATORS::MATM: // Dimensions sorted by Thomas later
            TypeCheckUtils::assertNumericType(lhsPrim);
            TypeCheckUtils::assertNumericType(rhsPrim);
            primType = TypeCheckUtils::getHighestType(lhsPrim, rhsPrim);
            this->type = TypeCheckUtils::makeMatrixType(lhsTy.getDimensions(), primType);
            break;
        case AST::BIN_OPERATORS::POW:
            TypeCheckUtils::assertNumericType(lhsPrim);
            if (TypeCheckUtils::isInt(rhsPrim)) {
                primType = Typing::PRIMITIVE::FLOAT;
                this->type = TypeCheckUtils::makeMatrixType(lhsTy.getDimensions(), primType);
            } else {
                TypeCheckUtils::wrongTypeError("Expected Int exponent", rhsPrim);
            }
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

/**
 * Function to determine whenever or not we execute the binary operation on the GPU or not
 * @param op
 * @return
 */
bool AST::BinaryExprNode::shouldExecuteGPU(Utils::IRContext* context, AST::BIN_OPERATORS op) const {
    // Define a lookup table for the operation complexity
    int entropy = 1;
    auto lhsMatNode = std::dynamic_pointer_cast<AST::ExprNode>(this->lhs);
    auto rhsMatNode = std::dynamic_pointer_cast<AST::ExprNode>(this->rhs);
    auto* lhsType = std::get_if<Typing::MatrixType>(&*lhsMatNode->type);
    auto* rhsType = std::get_if<Typing::MatrixType>(&*rhsMatNode->type);
    if (op == MATM) {
        // This is the only complex operation
        if (lhsType) {
            entropy = lhsType->dimensions[1];
        } else {
            if (context->compilerOptions->warningVerbosity == WARNINGS::ALL) {
                std::cout << "[Warning] - Matrix Multiplication entropy calculation failed - Using element wise entropy"
                          << std::endl;
            }
        }
    }
    entropy *= rhsType->getLength() * lhsType->getLength();
    int maxCPUEntropy = 400;  // 400 corresponds to 20x20 matrix
    return entropy >= maxCPUEntropy;
}

/**
 * Function computes the fast power of a value and an exponent using an iterative method described in
 * https://mathstats.uncg.edu/sites/pauli/112/HTML/secfastexp.html.
 *
 * This is O(ceil(log2(n))+1), rather than the O(n) multiplications required with naive method of multiplication power
 * @param context
 * @param lhs
 * @param rhs
 * @param isFloat
 * @param name
 * @return
 */
llvm::Value* AST::BinaryExprNode::applyPowerToOperands(Utils::IRContext* context, llvm::Value* lhs, llvm::Value* rhs,
                                                       const bool isFloat, const std::string& name) {
    auto tyInt = static_cast<llvm::Type*>(llvm::Type::getInt64Ty(context->module->getContext()));
    auto tyFloat64 = static_cast<llvm::Type*>(llvm::Type::getDoubleTy(context->module->getContext()));

    if (lhs->getType()->isIntegerTy() && rhs->getType()->isIntegerTy()) {
        llvm::Value* a = llvm::ConstantInt::get(tyInt, llvm::APInt(64, 1, true));
        // Copy over the values within our scope
        llvm::Value* c = lhs;
        llvm::Value* n = rhs;

        llvm::BasicBlock* negativeReverse =
            llvm::BasicBlock::Create(context->module->getContext(), "pow.neg", context->function);
        llvm::BasicBlock* powerLoopStart =
            llvm::BasicBlock::Create(context->module->getContext(), "pow.begin", context->function);
        llvm::BasicBlock* powerLoopEnd =
            llvm::BasicBlock::Create(context->module->getContext(), "pow.end", context->function);
        // Flip n if negative
        llvm::Value* negCmp = context->Builder->CreateICmpSLT(n, llvm::ConstantInt::get(tyInt, 0));
        context->Builder->CreateCondBr(negCmp, negativeReverse, powerLoopStart);
        context->Builder->SetInsertPoint(negativeReverse);
        {
            // n = (-1)*n
            n = context->Builder->CreateNeg(n);
        }
        // Power function loop
        context->Builder->SetInsertPoint(powerLoopStart);
        {
            // r = n mod 2
            llvm::Value* r = context->Builder->CreateSRem(n, llvm::ConstantInt::get(tyInt, 2));

            llvm::BasicBlock* powLoopInsideCondition =
                llvm::BasicBlock::Create(context->module->getContext(), "pow.cond", context->function);
            llvm::BasicBlock* powLoopInsideConditionEnd =
                llvm::BasicBlock::Create(context->module->getContext(), "pow.condEnd", context->function);

            // r == 1 condition
            auto* innerComparison = context->Builder->CreateICmpEQ(r, llvm::ConstantInt::get(tyInt, 1));
            // break if true
            context->Builder->CreateCondBr(innerComparison, powLoopInsideCondition, powLoopInsideConditionEnd);
            {
                context->Builder->SetInsertPoint(powLoopInsideCondition);
                a = context->Builder->CreateMul(a, c);
            }
            // n = n div 2
            n = context->Builder->CreateSDiv(n, llvm::ConstantInt::get(tyInt, 2));
            // c = c * c
            c = context->Builder->CreateMul(c, c);
            // break if n == 0
            auto* endCondition = context->Builder->CreateICmpEQ(n, llvm::ConstantInt::get(tyInt, 0));
            context->Builder->CreateCondBr(endCondition, powerLoopEnd, powerLoopStart);
        }
        // End of power loop
        context->Builder->SetInsertPoint(powerLoopEnd);
        llvm::BasicBlock* flipPos =
            llvm::BasicBlock::Create(context->module->getContext(), "pow.negFinish", context->function);
        llvm::BasicBlock* flipPosEnd =
            llvm::BasicBlock::Create(context->module->getContext(), "pow.negFinishEnd", context->function);
        // if negCmp is true, a = 1/a
        context->Builder->CreateCondBr(negCmp, flipPos, flipPosEnd);
        context->Builder->SetInsertPoint(flipPos);
        { a = context->Builder->CreateFDiv(llvm::ConstantFP::get(tyFloat64, 1), a); }
        context->Builder->SetInsertPoint(flipPosEnd);
        return a;
    } else if (lhs->getType()->isFloatTy() && rhs->getType()->isIntegerTy()) {
        // For floating point numbers
        llvm::Value* a = llvm::ConstantFP::get(tyFloat64, 1);
        // Copy over the values within our scope
        llvm::Value* c = lhs;
        llvm::Value* n = rhs;

        llvm::BasicBlock* negativeReverse =
            llvm::BasicBlock::Create(context->module->getContext(), "pow.neg", context->function);
        llvm::BasicBlock* powerLoopStart =
            llvm::BasicBlock::Create(context->module->getContext(), "pow.begin", context->function);
        llvm::BasicBlock* powerLoopEnd =
            llvm::BasicBlock::Create(context->module->getContext(), "pow.end", context->function);
        // Flip n if negative
        llvm::Value* negCmp = context->Builder->CreateICmpSLT(n, llvm::ConstantInt::get(tyInt, 0));
        context->Builder->CreateCondBr(negCmp, negativeReverse, powerLoopStart);
        context->Builder->SetInsertPoint(negativeReverse);
        {
            // n = (-1)*n
            n = context->Builder->CreateNeg(n);
        }
        // Power function loop
        context->Builder->SetInsertPoint(powerLoopStart);
        {
            // r = n mod 2
            llvm::Value* r = context->Builder->CreateSRem(n, llvm::ConstantInt::get(tyInt, 2));

            llvm::BasicBlock* powLoopInsideCondition =
                llvm::BasicBlock::Create(context->module->getContext(), "pow.cond", context->function);
            llvm::BasicBlock* powLoopInsideConditionEnd =
                llvm::BasicBlock::Create(context->module->getContext(), "pow.condEnd", context->function);

            // r == 1 condition (ordered fp)
            auto* innerComparison = context->Builder->CreateICmpEQ(r, llvm::ConstantInt::get(tyInt, 1));
            // break if true
            context->Builder->CreateCondBr(innerComparison, powLoopInsideCondition, powLoopInsideConditionEnd);
            {
                context->Builder->SetInsertPoint(powLoopInsideCondition);
                a = context->Builder->CreateFMul(a, c);
            }
            // n = n div 2 (n remains int)
            n = context->Builder->CreateSDiv(n, llvm::ConstantInt::get(tyInt, 2));
            // c = c * c
            c = context->Builder->CreateFMul(c, c);
            // break if n == 0
            auto* endCondition = context->Builder->CreateICmpEQ(n, llvm::ConstantInt::get(tyInt, 0));
            context->Builder->CreateCondBr(endCondition, powerLoopEnd, powerLoopStart);
        }
        // End of power loop
        context->Builder->SetInsertPoint(powerLoopEnd);
        llvm::BasicBlock* flipPos =
            llvm::BasicBlock::Create(context->module->getContext(), "pow.negFinish", context->function);
        llvm::BasicBlock* flipPosEnd =
            llvm::BasicBlock::Create(context->module->getContext(), "pow.negFinishEnd", context->function);
        // if negCmp is true, a = 1/a
        context->Builder->CreateCondBr(negCmp, flipPos, flipPosEnd);
        context->Builder->SetInsertPoint(flipPos);
        { a = context->Builder->CreateFDiv(llvm::ConstantFP::get(tyFloat64, 1), a); }
        context->Builder->SetInsertPoint(flipPosEnd);
        return a;
    } else {
        throw std::runtime_error("Unsupported exponent or base: Supported operations: Integer^Integer, Float^Integer");
    }
}

const char* op_name(BIN_OPERATORS i) { return BIN_OP_ENUM_STRING[(int)i]; }

std::string AST::BinaryExprNode::toTree(const std::string& prefix, const std::string& childPrefix) const {
    using namespace Tree;
    std::string str{prefix + std::string{"Binary Expression: "} + op_name(this->op) + "\n"};
    str += lhs->toTree(childPrefix + T, childPrefix + I);
    str += rhs->toTree(childPrefix + L, childPrefix + B);
    return str;
}
}  // namespace AST