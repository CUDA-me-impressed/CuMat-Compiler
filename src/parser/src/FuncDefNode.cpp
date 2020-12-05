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
    llvm::FunctionType* ft =
        llvm::FunctionType::get(std::get<Typing::MatrixType>(*this->returnType).getLLVMType(context), argTypes, false);
    llvm::Function* func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, this->funcName, context->module);

    // Store within the symbol table
    Utils::funcTable[this->funcName][typesRaw] = func;

    // For this function, we need a new BasicBlock structure
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(context->module->getContext(), "func" + this->funcName, func,
                                                    context->Builder->GetInsertBlock());
    context->Builder->SetInsertPoint(bb);

    // TODO: MOVE TO BLOCK CODEGEN
    // TODO: Deal with the assignments

    // CodeGen for Block
    llvm::Value* retVal = block->codeGen(context);
    context->Builder->CreateRet(retVal);
}