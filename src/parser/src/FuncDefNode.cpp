#include "FuncDefNode.hpp"

llvm::Value* AST::FuncDefNode::codeGen(Utils::IRContext* context) {
    // Let us generate a new function -> We will first generate the function argument types
    std::vector<llvm::Type*> argTypes;
    std::vector<std::shared_ptr<Typing::Type>> typesRaw;
    for (const auto& typeNamePair : this->parameters) {
        typesRaw.push_back(typeNamePair.second);
        argTypes.push_back(std::get<Typing::MatrixType>(*typeNamePair.second).getLLVMType(context));
    }

    // Get out the type and create a function
    auto mt = std::get<Typing::MatrixType>(*this->returnType);
    //TODO: Figure out if we should be passing a pointer or the whole struct
    auto* mtType = mt.getLLVMType(context)->getPointerTo();
    llvm::FunctionType* ft = llvm::FunctionType::get(mtType, argTypes, false);
    llvm::Function* func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, this->funcName, context->module);

    // Store within the symbol table
    Utils::funcTable[this->funcName][typesRaw] = func;
    context->function = func;

    auto* funcRet = this->block->codeGen(context);
    return funcRet;
}