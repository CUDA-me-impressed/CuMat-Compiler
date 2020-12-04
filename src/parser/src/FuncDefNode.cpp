#include "FuncDefNode.hpp"

llvm::Value* AST::FuncDefNode::codeGen(Utils::IRContext* context) {
    // Let us generate a new function -> We will first generate the function argument types
    std::vector<llvm::Type*> argTypes;
    for (const auto& typeNamePair : this->parameters) {
        argTypes.push_back(std::get<Typing::MatrixType>(*typeNamePair.second).getLLVMType(context->module));
    }
    llvm::FunctionType* ft = llvm::FunctionType::get(
        std::get<Typing::MatrixType>(*this->returnType).getLLVMType(context->module), argTypes, false);
    llvm::Function* func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, this->funcName, context->module);
    // For this function, we need a new BasicBlock structure
    llvm::BasicBlock* bb =
        llvm::BasicBlock::Create(context->module->getContext(), "func" + this->funcName,
                                 context->Builder->GetInsertBlock()->getParent(), context->Builder->GetInsertBlock());
    context->Builder->SetInsertPoint(bb);
    // Deal with input variables

    // TODO: Deal with the assignments

    // Generate Return statement code
    llvm::Value* retVal = returnExpr->codeGen(context);
    context->Builder->CreateRet(retVal);
}