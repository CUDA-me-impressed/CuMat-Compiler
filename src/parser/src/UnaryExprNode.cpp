#include "UnaryExprNode.hpp"

#include "CodeGenUtils.hpp"
#include "MatrixNode.hpp"

llvm::Value* AST::UnaryExprNode::codeGen(llvm::Module* module, llvm::IRBuilder<>* Builder, llvm::Function* fp) {
    // opval should be an evaluated matrix for which we can create a new matrix
    llvm::Value* opVal = this->operand->codeGen(module, Builder, fp);
    // We go through and apply the relevant unary operator to each element of
    // the matrix
    auto matType = std::dynamic_pointer_cast<AST::MatrixNode>(this->operand);
    llvm::Type* ty = matType->getLLVMType(module);
    auto dimension = matType->getDimensions();
    auto newMatAlloc = Utils::generateMatrixAllocation(ty, dimension, Builder);
    // We generate the operations sequentially
    // TODO: Add Kernel call for nvptx
    recursiveUnaryGeneration(op, module, Builder, ty, newMatAlloc, opVal, dimension);
    return opVal;
}

/**
 * Generates LLVM IR for the Unary Expression via a recursive pass over each
 * element of the matrix in a column major systematic search. This will output
 * code that NVPTX should be able to identifiy and use to generate PTX compliant
 * code optimised for CUDA.
 *
 * Note, final two paramaters, index and prevDim can be excluded as they are
 * used within the function to recursively generate the index offset. By default
 * they are set to the identity element (1) and will not impact the result.
 * @param op
 * @param module
 * @param Builder
 * @param ty
 * @param matAlloc
 * @param opVal
 * @param dimension
 * @param index
 * @param prevDim
 */
void AST::UnaryExprNode::recursiveUnaryGeneration(const UNA_OPERATORS& op, llvm::Module* module,
                                                  llvm::IRBuilder<>* Builder, llvm::Type* ty,
                                                  llvm::AllocaInst* matAlloc, llvm::Value* opVal,
                                                  std::vector<int> dimension, int index, int prevDim) {
    // Store of a type for the matrix (only need this for the final pass)
    llvm::ArrayType* matType;
    if (dimension.size() == 1) {
        matType = llvm::ArrayType::get(ty, index * dimension.at(0));
    }

    for (int i = 0; i < dimension.at(0); i++) {
        // If we have more than one dimension, we need to explore the matrix
        // more
        if (dimension.size() > 1) {
            // Create a new dimension vector with this dimension removed
            std::vector<int> subDimension(dimension.begin() + 1, dimension.end());
            recursiveUnaryGeneration(op, module, Builder, ty, matAlloc, opVal, subDimension, (index * prevDim) + i,
                                     dimension.at(0));
        } else {
            // At this point, the element that will be contained is the most raw
            // llvm value, indexed at position
            // TODO: Offset needs to work with non-64 bit variables
            auto zero = llvm::ConstantInt::get(module->getContext(), llvm::APInt(64, 0, true));
            auto indexVal = llvm::ConstantInt::get(module->getContext(), llvm::APInt(64, index, true));
            // Pointer to the index within IR
            auto ptrOld =
                llvm::GetElementPtrInst::Create(matType, opVal, {zero, indexVal}, "", Builder->GetInsertBlock());
            auto ptrNew =
                llvm::GetElementPtrInst::Create(matType, matAlloc, {zero, indexVal}, "", Builder->GetInsertBlock());
            // Generate the code for each valid operation and type
            /*TODO: Probably needs syntax checking, leaving this for someone
             * with a better understanding of programming language theory
             * */
            switch (this->op) {
                case NEG: {
                    if (ty->isIntegerTy()) {
                        auto neg =
                            llvm::BinaryOperator::CreateNeg(Builder->CreateLoad(ptrOld), "", Builder->GetInsertBlock());
                        // Insert
                        llvm::StoreInst(neg, ptrNew, false, Builder->GetInsertBlock());
                    } else if (ty->isFloatTy()) {
                        auto neg = llvm::BinaryOperator::CreateFNeg(Builder->CreateLoad(ptrOld), "",
                                                                    Builder->GetInsertBlock());
                        llvm::StoreInst(neg, ptrNew, false, Builder->GetInsertBlock());
                    }
                    break;
                }
                case LNOT: {
                    // TODO: Linear not? Can someone check on this? Same for
                    // BNOT
                    llvm::BinaryOperator::CreateNot(Builder->CreateLoad(ptrOld), "", Builder->GetInsertBlock());
                    break;
                }
                case BNOT: {
                    llvm::BinaryOperator::CreateNot(Builder->CreateLoad(ptrOld), "", Builder->GetInsertBlock());
                    break;
                }
            }
        }
    }
}