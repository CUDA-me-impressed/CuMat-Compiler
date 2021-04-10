#include "AssignmentNode.hpp"
#include "TypeCheckingUtils.hpp"

#include <iostream>
#include <utility>
#include <numeric>

llvm::Value* AST::AssignmentNode::codeGen(Utils::IRContext* context) {
    // Generate LLVM value for the rval expression
    llvm::Value* rValLLVM = this->rVal->codeGen(context);

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
        if (isFunction) {
            // Cannot decompose function
            std::exit(9);
        } else if (rValTy != nullptr) {
            this->lVal->semanticPass(context, rValTy->getPrimitiveType());
        }
    } else {
        if (context->semanticSymbolTable->inVarTable(this->name)) {
            TypeCheckUtils::alreadyDefinedError(this->name);
        }
        if (isFunction) {
            // store a function type as a variable
            AST::VariableNode varNode = std::get_if<AST::VariableNode>(&*this->rVal);
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

    // Create the l and r value types for the decomposition
    std::shared_ptr<Typing::MatrixType> lValMatType = std::make_shared<Typing::MatrixType>();
    std::shared_ptr<Typing::MatrixType> rValMatType = std::make_shared<Typing::MatrixType>();
    // Set the dimensionality / rank of the types
    lValMatType->rank = matType->rank - 1;
    rValMatType->rank = matType->rank - 1;
    lValMatType->dimensions = std::vector<uint>(matType->dimensions.begin(), matType->dimensions.end() - 1);
    rValMatType->dimensions = matType->dimensions;
    rValMatType->dimensions.insert(rValMatType->dimensions.begin(), rValMatType->dimensions.front() - 1);
    // Create the matricies in LLVM to store these l/r vals
    auto* lValMatAlloc = Utils::createMatrix(context, *lValMatType);
    auto* rValMatAlloc = Utils::createMatrix(context, *rValMatType);
    // Get mat record out
    auto lValMatRecord = Utils::getMatrixFromPointer(context, lValMatAlloc);
    auto rValMatRecord = Utils::getMatrixFromPointer(context, rValMatAlloc);
    auto matRecord = Utils::getMatrixFromPointer(context, matHeader);
    // Calculate offset of the rVal data address

    llvm::Value* rValDataPtr = context->Builder->CreateGEP(matRecord.dataPtr, lValMatRecord.numBytes, "rValOffset");
    // Point both of the data pointers to the correct locations
    Utils::insertValueAtPointerOffset(context, lValMatRecord.dataPtr, 0, matRecord.dataPtr, false);
    Utils::insertValueAtPointerOffset(context, rValMatRecord.dataPtr, 0, rValDataPtr, false);

    // Handle assignment symbol table code
    if (!context->symbolTable->inSymbolTable(this->name, context->symbolTable->getCurrentFunction())) {
        // Something has gone wrong during the parse stage and we have not added the symbol into the table
        // Raising a warning!
        if(context->compilerOptions->warningVerbosity == WARNINGS::ALL) {
            std::cout << "[Internal Warning] Symbol " << this->name
                      << " was not found within the symbol"
                         " table. Created during codegen"
                      << std::endl;
        }
        // No typing information can be inferred at this stage (nullptr) - Can and will cause issues hence the warning
        context->symbolTable->setValue(nullptr, lValMatAlloc, this->name, context->symbolTable->getCurrentFunction());
    } else {
        context->symbolTable->updateValue(lValMatAlloc, this->name, context->symbolTable->getCurrentFunction());
    }

    // Handle any of the sub decompositions, these are not returned as we are not reducing the dimensions of them and we
    // only pick the first variable assigned, but they will be present within the symbol table
    std::shared_ptr<DecompNode> nestedDecomp = *std::get_if<std::shared_ptr<DecompNode>>(&decomp->rVal);
    if (nestedDecomp) {
        decompAssign(context, nestedDecomp, lValMatAlloc);
    }
    return lValMatAlloc;  // Dunno, seems to be what I would want, maybe change?
}
