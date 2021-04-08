#include "ProgramNode.hpp"

llvm::Value* AST::ProgramNode::codeGen(Utils::IRContext* context) {
    auto* mtType = llvm::Type::getVoidTy(context->module->getContext());
    std::vector<llvm::Type*> argTypes({
        llvm::Type::getInt64PtrTy(context->module->getContext()),
        llvm::Type::getInt64PtrTy(context->module->getContext()),
        llvm::Type::getInt64PtrTy(context->module->getContext()),
        llvm::Type::getInt64Ty(context->module->getContext())
    });

    llvm::FunctionType* ft = llvm::FunctionType::get(mtType, argTypes, false);
    llvm::Function *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "CuMatAddMatrixI", context->module);
    return Node::codeGen(context);
}
