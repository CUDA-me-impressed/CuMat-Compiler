#include "AssignmentNode.hpp"
#include "TypeCheckingUtils.hpp"

#include <iostream>
#include <numeric>

#include "DimensionsSymbolTable.hpp"
#include "TreePrint.hpp"

llvm::Value* AST::AssignmentNode::codeGen(Utils::IRContext* context) {
    // Generate LLVM value for the rval expression
    llvm::Value* rValLLVM = this->rVal->codeGen(context);

    // Ensure that matrix literals are upcast
    if(auto* rValType = std::get_if<Typing::MatrixType>(&*rVal->type)){
        if (this->rVal->isLiteralNode()) {
            rValLLVM = Utils::upcastLiteralToMatrix(context, *rValType, rValLLVM);
        }
    }

    // Handle decomposition
    if (this->lVal) {
        return decompAssign(context, this->lVal, rValLLVM);
    } else {
        // Normal variable assignment
        if (!context->symbolTable->inSymbolTable(this->name, context->symbolTable->getCurrentFunction())) {
            // Something has gone wrong during the parse stage and we have not added the symbol into the table
            // Raising a warning!
            if (context->compilerOptions->warningVerbosity == WARNINGS::ALL) {
                std::cout << "[Internal Warning] Symbol " << this->name
                          << " was not found within the symbol"
                             " table. Created during codegen"
                          << std::endl;
            }
            // No typing information can be inferred at this stage (nullptr) - Can and will cause issues hence the
            // warning
            context->symbolTable->setValue(nullptr, rValLLVM, this->name, context->symbolTable->getCurrentFunction());
        } else {
            context->symbolTable->updateValue(rValLLVM, this->name, context->symbolTable->getCurrentFunction());
        }
        return rValLLVM;
    }
}

void AST::AssignmentNode::semanticPass(Utils::IRContext* context) {
    this->rVal->semanticPass(context);
    auto rValTy = std::get_if<Typing::MatrixType>(this->rVal->type.get());
    auto rValFty = std::get_if<Typing::FunctionType>(this->rVal->type.get());
    bool isFunction = rValFty != nullptr;

    if (this->lVal != nullptr) {
        // Check if decomposition is taking place
        if (isFunction) {
            // Cannot decompose function
            TypeCheckUtils::decompError();
        } else if (rValTy != nullptr) {
            // Else, call semantic pass on rVal passing through the primitive type
            this->lVal->semanticPass(context, rValTy->getPrimitiveType());
        }
    } else {
        // In this branch, only the `name` attribute is defined, signalling simple assignment
        if (context->semanticSymbolTable->inVarTable(this->name)) {
            // Error if the variable name is already in use
            TypeCheckUtils::alreadyDefinedError(this->name, true);
        }
        if (isFunction) {
            // Store a function type as a variable
            AST::VariableNode varNode = *dynamic_cast<AST::VariableNode*>(this->rVal.get());
            // Concatenate the namespace into a single string
            std::string nameSpace = std::accumulate(varNode.namespacePath.begin(), varNode.namespacePath.end(), std::string(""));
            context->semanticSymbolTable->storeVarType(this->name, nullptr, nameSpace, varNode.name);
        } else {
            context->semanticSymbolTable->storeVarType(this->name, this->rVal->type);
        }

        if (context->symbolTable->inSymbolTable(this->name, context->symbolTable->getCurrentFunction())) {
            throw std::runtime_error("Attempting to redefine variable: " + this->name);
        }
        context->symbolTable->setValue(this->rVal->type, nullptr, this->name,
                                       context->symbolTable->getCurrentFunction());
    }
}

llvm::Value* AST::AssignmentNode::decompAssign(Utils::IRContext* context, std::shared_ptr<DecompNode> decomp,
                                               llvm::Value* matHeader) {
    // Get the type for the original value
    auto matType = std::get_if<Typing::MatrixType>(&*this->rVal->type);
    if (!matType) {
        if (context->compilerOptions->warningVerbosity == WARNINGS::ALL) {
            std::cout << "[Internal Warning] Cannot find type information for rVal with variable " << name << std::endl;
        }
        // Attempt correction
    }
    auto nestedDecomp = std::get_if<std::shared_ptr<DecompNode>>(&decomp->rVal);

        // Create the l and r value types for the decomposition
        std::shared_ptr<Typing::MatrixType>
            lValMatType = std::make_shared<Typing::MatrixType>();
    std::shared_ptr<Typing::MatrixType> rValMatType = std::make_shared<Typing::MatrixType>();
    // Set the dimensionality / rank of the types
    lValMatType->rank = matType->rank != 1 ? matType->rank - 1 : 1;
    lValMatType->dimensions =
        std::vector<uint>(matType->dimensions.begin(), matType->dimensions.begin() + lValMatType->rank);
    lValMatType->primType = matType->getPrimitiveType();

    rValMatType->rank = matType->rank;
    rValMatType->dimensions = std::vector<uint>(matType->dimensions.begin(), matType->dimensions.end());
    rValMatType->primType = matType->getPrimitiveType();
    // Create the matricies in LLVM to store these l/r vals
    auto* lValMatAlloc = Utils::createMatrix(context, *lValMatType);
    auto* rValMatAlloc = Utils::createMatrix(context, *rValMatType);
    // Get mat record out
    auto lValMatRecord = Utils::getMatrixFromPointer(context, lValMatAlloc);
    auto rValMatRecord = Utils::getMatrixFromPointer(context, rValMatAlloc);

    auto matRecord = Utils::getMatrixFromPointer(context, matHeader);
    // Calculate offset of the rVal data address

    //    llvm::Value* rValDataPtr = context->Builder->CreateGEP(matRecord.dataPtr, lValMatRecord.numBytes,
    //    "rValOffset");
    // Point both of the data pointers to the correct locations
    auto* i32Ty = llvm::Type::getInt32Ty(context->module->getContext());
    auto dataPtr = Utils::getPointerAddressFromOffset(context, matRecord.dataPtr, llvm::ConstantInt::get(i32Ty, 0));
    auto secondDimDataPtr = Utils::getPointerAddressFromOffset(context, matRecord.dataPtr,
                                                               llvm::ConstantInt::get(i32Ty, matType->dimensions[0]));
    auto numElements =
        std::accumulate(lValMatType->dimensions.begin(), lValMatType->dimensions.end(), 1, std::multiplies<>{});
    llvm::Value* firstSize =
        llvm::ConstantExpr::getMul(llvm::ConstantInt::get(i32Ty, numElements), llvm::ConstantInt::get(i32Ty, 8));
    llvm::Value* secondSize =
        llvm::ConstantExpr::getMul(llvm::ConstantExpr::getMul(llvm::ConstantInt::get(i32Ty, matType->dimensions[0]),
                                                              llvm::ConstantInt::get(i32Ty, matType->rank - 1)),
                                   llvm::ConstantInt::get(i32Ty, 8));
    context->Builder->CreateMemCpy(lValMatRecord.dataPtr,
                                   lValMatRecord.dataPtr->getPointerAlignment(context->module->getDataLayout()),
                                   dataPtr, dataPtr->getPointerAlignment(context->module->getDataLayout()), firstSize);
    context->Builder->CreateMemCpy(rValMatRecord.dataPtr,
                                   rValMatRecord.dataPtr->getPointerAlignment(context->module->getDataLayout()),
                                   secondDimDataPtr, secondDimDataPtr->getPointerAlignment(context->module->getDataLayout()), secondSize);

    // Handle assignment symbol table code
    if (!context->symbolTable->inSymbolTable(decomp->lVal, context->symbolTable->getCurrentFunction())) {
        // Something has gone wrong during the parse stage and we have not added the symbol into the table
        // Raising a warning!
        if (context->compilerOptions->warningVerbosity == WARNINGS::ALL) {
            std::cout << "[Internal Warning] Symbol " << this->name
                      << " was not found within the symbol"
                         " table. Created during codegen"
                      << std::endl;
        }
        // No typing information can be inferred at this stage (nullptr) - Can and will cause issues hence the warning
        context->symbolTable->setValue(nullptr, lValMatAlloc, decomp->lVal, context->symbolTable->getCurrentFunction());
    } else {
        context->symbolTable->updateValue(lValMatAlloc, decomp->lVal, context->symbolTable->getCurrentFunction());
    }

    // Handle assignment symbol table code
    if(!nestedDecomp) {
        auto rValName = *std::get_if<std::string>(&decomp->rVal);

        if (!context->symbolTable->inSymbolTable(rValName, context->symbolTable->getCurrentFunction())) {
            // Something has gone wrong during the parse stage and we have not added the symbol into the table
            // Raising a warning!
            if (context->compilerOptions->warningVerbosity == WARNINGS::ALL) {
                std::cout << "[Internal Warning] Symbol " << this->name
                          << " was not found within the symbol"
                             " table. Created during codegen"
                          << std::endl;
            }
            // No typing information can be inferred at this stage (nullptr) - Can and will cause issues hence the warning
            context->symbolTable->setValue(nullptr, rValMatAlloc, rValName,
                                           context->symbolTable->getCurrentFunction());
        } else {
            context->symbolTable->updateValue(rValMatAlloc, rValName, context->symbolTable->getCurrentFunction());
        }
    }
    // Handle any of the sub decompositions, these are not returned as we are not reducing the dimensions of them and we
    // only pick the first variable assigned, but they will be present within the symbol table

    if (nestedDecomp) {
        decompAssign(context, *nestedDecomp, lValMatAlloc);
    }
    return lValMatAlloc;  // Dunno, seems to be what I would want, maybe change?
}

std::string AST::AssignmentNode::toTree(const std::string& prefix, const std::string& childPrefix) const {
    using namespace Tree;
    std::string str{prefix + std::string{"Assignment\n"}};
    str += lVal->toTree(childPrefix + T, childPrefix + I);
    str += rVal->toTree(childPrefix + L, childPrefix + B);
    return str;
}
void AST::AssignmentNode::dimensionPass(Analysis::DimensionSymbolTable* nt) {
    rVal->dimensionPass(nt);
    if (!this->name.empty()) {
        nt->add_node(this->name, rVal->type);
    } else {
        if (auto* t = std::get_if<Typing::MatrixType>(rVal->type.get())) {
            lVal->dimensionPass(nt, *t);
        }
    }
}
