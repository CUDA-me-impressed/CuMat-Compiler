#include "FunctionExprNode.hpp"
#include "CodeGenUtils.hpp"

#include <map>

llvm::Value* AST::FunctionExprNode::codeGen(Utils::IRContext* context) {
    // We will attempt to retrieve the function object from symbol table via reference to name + arg type

    // We need to generate types and codegen for arguments
    if(!Utils::funcTable.contains(this->funcName)) return nullptr;  // TODO: Handle graceful error message
    std::map<std::vector<std::shared_ptr<Typing::Type>>, llvm::Function*> funcArgParams = Utils::funcTable[this->funcName];

    std::vector<std::shared_ptr<Typing::Type>> argTypesRaw;
    for (const auto& typeNamePair : this->args) {
        argTypesRaw.push_back(typeNamePair->type);
    }
    if(!funcArgParams.contains(argTypesRaw)) return nullptr; // TODO: Handle graceful error message
    auto* func = funcArgParams[argTypesRaw];

    if(func->arg_size() != this->args.size()) return nullptr; // TODO: Handle graceful error message

    // Generate return values for each of the evaluations of the function
    std::vector<llvm::Value*> argVals;
    for (const auto& arg : args) {
        argVals.push_back(arg->codeGen(context));
    }
    // Call the function with values
    auto callRet = context->Builder->CreateCall(func, argVals, "calltmp");
    return callRet;
}