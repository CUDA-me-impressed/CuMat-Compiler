#include "FunctionExprNode.hpp"

#include <map>

#include "CodeGenUtils.hpp"
#include "DimensionsSymbolTable.hpp"
#include "TreePrint.hpp"

llvm::Value* AST::FunctionExprNode::codeGen(Utils::IRContext* context) {
    // We will attempt to retrieve the function object from symbol table via reference to name + arg type

    // We need to generate types and codegen for arguments
    const std::string funcName = this->nonAppliedFunction->literalText;
    if (!context->symbolTable->isFunctionDefined(funcName)) {
        // If we failed to identify this issue until here, we fucked up internally
        // Semantic checking during the parse tree Antlr generation NEEDS to pick this up!
        throw std::runtime_error("[Internal error] Function [" + funcName +
                                 "] was not defined before called in expr node");
    }

    std::vector<std::shared_ptr<Typing::Type>> argTypesRaw;
    for (const auto& typeNamePair : this->args) {
        argTypesRaw.push_back(typeNamePair->type);
    }

    //    if(!context->symbolTable->isFunctionDefinedParam(funcName, argTypesRaw)){
    //        throw std::runtime_error("[Internal error] Function [" + funcName +
    //                                 "] defined however parameters do not match");
    //    }
    auto* func = context->symbolTable->getFunction(funcName, argTypesRaw).func;

    if (func->arg_size() != this->args.size()) return nullptr;  // TODO: Handle graceful error message

    // Generate return values for each of the evaluations of the function
    std::vector<llvm::Value*> argVals;
    for (const auto& arg : args) {
        argVals.push_back(arg->codeGen(context));
    }
    // Call the function with values
    auto callRet = context->Builder->CreateCall(func, argVals, "calltmp");
    return callRet;
}

void AST::FunctionExprNode::semanticPass(Utils::IRContext* context) {
    this->nonAppliedFunction->semanticPass(context);
    for (auto const& arg : this->args) arg->semanticPass(context);
    // TODO: Check that types align with the argument types specified in the symbol table
}

std::string AST::FunctionExprNode::toTree(const std::string& prefix, const std::string& childPrefix) const {
    using namespace Tree;
    std::string str{prefix + std::string{"Function Application"}};
    if (!funcName.empty()) {
        str += ": " + funcName;
    }
    for (auto const& node : this->args) {
        if (&node != &this->args.back()) {
            str += node->toTree(childPrefix + T, childPrefix + I);
        } else
            str += node->toTree(childPrefix + L, childPrefix + B);
    }
    return str;
}

void AST::FunctionExprNode::dimensionPass(Analysis::DimensionSymbolTable* nt) {
    for (auto& a : args) {
        a->dimensionPass(nt);
    }
    if (std::holds_alternative<Typing::FunctionType>(*nonAppliedFunction->type)) {
        if (nt->search_impl(nonAppliedFunction->literalText)) {
        }
    }
}
