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
    // TODO: Figure out if we should be passing a pointer or the whole struct
    auto* mtType = mt.getLLVMType(context)->getPointerTo();
    llvm::FunctionType* ft = llvm::FunctionType::get(mtType, argTypes, false);
    llvm::Function* func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, this->funcName, context->module);

    // Store within the symbol table
    context->symbolTable->addNewFunction(funcName, typesRaw);  // This should be done within the semantic pass
    context->symbolTable->setFunctionData(funcName, typesRaw, func);
    context->function = func;

    auto* funcRet = this->block->codeGen(context);

    // Pop the function as we leave the definition of the code
    context->symbolTable->escapeFunction();

    // We need to generate meta-data for NVPTX
    // This is 100% stolen from https://stackoverflow.com/questions/40082378/how-to-generate-metadata-for-llvm-ir
    // as it is someone asking how to do this exact problem :)

    // Vector to store the tuple operations
    llvm::SmallVector<llvm::Metadata*, 3> ops;
    // We reference the type first from the global llvm symbol lookup rather than internal
    // as then we can guarantee we haven't messed up thus far!
    llvm::GlobalValue * funcGlob = context->module->getNamedValue(this->funcName);
    if(!funcGlob){
        throw std::runtime_error("[Internal Error] Could not find function to generate metadata for!");
    }

    // Push the function reference
    ops.push_back(llvm::ValueAsMetadata::getConstant(funcGlob));
    // Push the type of the function (device or kernel)
    ops.push_back(llvm::MDString::get(context->module->getContext(), "kernel"));

    // We need an i64Ty to tell nvptx what API to use (I think)
    llvm::Type *i64ty = llvm::Type::getInt64Ty(context->module->getContext());
    llvm::Constant *one = llvm::ConstantInt::get(i64ty, 1);
    ops.push_back(llvm::ValueAsMetadata::getConstant(one));

    // Generate the tuple with operands and attach it to the function as metadata
    auto *node = llvm::MDTuple::get(context->module->getContext(), ops);
    func->setMetadata("nvptx", node);

    return funcRet;
}