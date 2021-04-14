#include "DecompNode.hpp"

#include <variant>

#include "DimensionsSymbolTable.hpp"

llvm::Value* AST::DecompNode::codeGen(Utils::IRContext* context) {
    //    if (auto variableName = std::get_if<std::string>(&rVal)) {
    //        // Get out the dimensionality -> We assume we have a matrix in the rval
    //        auto varRecord = context->symbolTable->getValue(std::string(reinterpret_cast<const char*>(variableName)),
    //                                                        context->symbolTable->getCurrentFunction());
    //        auto matrixRecord = Utils::getMatrixFromPointer(context, varRecord->llvmVal);
    //        // Resultant rank is lower than the previous one
    //        llvm::Value* resultRank = context->Builder->CreateSub(matrixRecord.rank, Utils::getValueFromLLVM(context,
    //        1, Typing::PRIMITIVE::INT, false));
    //
    //        // Create a store matrix for the decomposition
    //        Typing::MatrixType* matType = std::get_if<Typing::MatrixType>(&*varRecord->type);
    //        matType->rank = matType->rank - 1;  // Subtract the rank
    //        matType->dimensions.pop_back();     // Remove the last dimension as we no longer need it
    //
    //        auto* matAlloc = Utils::createMatrix(context, *matType);
    //        auto* dataPtr = Utils::getValueFromPointerOffset(context, matAlloc, 0, "matArrPtr");
    //        auto ty = static_cast<llvm::Type*>(llvm::Type::getInt64Ty(context->module->getContext()));
    //
    //        // Allocate space for the new sub-matrix
    //        int newSize = matType->getLength() / *matType->dimensions.end();
    //        llvm::ArrayType* newMatType = llvm::ArrayType::get(ty, newSize);
    //
    //        // Allocate data for the new memory
    //        auto* matDataAlloc = context->Builder->CreateAlloca(newMatType, 0, nullptr, "matVar");
    //
    //        // Create LLVM Memory copy as we have row major representation
    //        context->Builder->CreateMemCpy(matDataAlloc, 0, matrixRecord.dataPtr, 0, newSize, false);
    //
    //        // Insert the copied data into the pointer at the offset from the initial matrix address
    //        Utils::insertValueAtPointerOffset(context, Utils::getMatrixFromPointer(context, matAlloc).dataPtr, 0,
    //        matDataAlloc);
    //
    //        return matAlloc;
    //    } else if(std::shared_ptr<DecompNode> decompNode = *std::get_if<std::shared_ptr<DecompNode>>(&rVal)){
    //        // Calculate any decomposition code generation before hand
    //        llvm::Value* inMatrix = decompNode->codeGen(context);
    //        auto matrixRecord = Utils::getMatrixFromPointer(context, inMatrix);
    //        std::shared_ptr<Typing::MatrixType> matType = std::make_shared<Typing::MatrixType>();
    //
    //        return nullptr;
    //    }
    return nullptr;
}

void AST::DecompNode::semanticPass(Utils::IRContext* context) {
    if (this->rVal.index() == 1) {
        std::shared_ptr<AST::DecompNode> child = std::get<std::shared_ptr<AST::DecompNode>>(this->rVal);
    }
}
void AST::DecompNode::dimensionPass(Analysis::DimensionSymbolTable* nt, Typing::MatrixType& type) {
    Typing::MatrixType lType = type;
    Typing::MatrixType rType = type;
    lType.rank--;
    lType.dimensions.pop_back();
    rType.dimensions.back()--;

    nt->add_node(this->lVal, std::move(std::make_shared<Typing::Type>(lType)));
    std::string* rName = std::get_if<std::string>(&this->rVal);
    std::shared_ptr<DecompNode>* rNode = std::get_if<std::shared_ptr<DecompNode>>(&this->rVal);
    if (rName) {
        nt->add_node(this->lVal, std::move(std::make_shared<Typing::Type>(rType)));
    } else if (rNode) {
        (*rNode)->dimensionPass(nt, rType);
    }
}
