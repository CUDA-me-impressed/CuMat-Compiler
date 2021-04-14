#include "FunctionExprNode.hpp"

#include <map>
#include <iostream>
#include <numeric>
#include <vector>

#include "CodeGenUtils.hpp"
#include "VariableNode.hpp"
#include "TypeCheckingUtils.hpp"
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
    // Check type of nonAppliedFunction variable
    AST::VariableNode nonAppliedFunc;
    try {
        nonAppliedFunc = *dynamic_cast<AST::VariableNode*>(this->nonAppliedFunction.get());
    } catch (std::bad_cast b) {
        std::cerr << "Only Variables can be called as functions" << std::endl;
        std::exit(TypeCheckUtils::ErrorCodes::FUNCTION_ERROR);
    }
    // If slicing exists, throw an error
    if (nonAppliedFunc.variableSlicing != nullptr) {
        std::cerr << "Cannot slice a function" << std::endl;
        std::exit(TypeCheckUtils::ErrorCodes::FUNCTION_ERROR);
    }
    // Check that function exists in the symbol table
    std::string nameSpace = std::accumulate(nonAppliedFunc.namespacePath.begin(), nonAppliedFunc.namespacePath.end(), std::string(""));
    if (!context->semanticSymbolTable->inFuncTable(nonAppliedFunc.name, nameSpace)) {
        TypeCheckUtils::notDefinedError(nonAppliedFunc.name);
    }
    auto funcType = std::get_if<Typing::FunctionType>(context->semanticSymbolTable->getFuncType(nonAppliedFunc.name, nameSpace).get());

    // Get the argument types for the function
    std::vector<std::shared_ptr<Typing::Type>> argTypes;
    for (auto const& arg : this->args) {
        arg->semanticPass(context);
        argTypes.push_back(std::move(arg->type));
        // Check that the argument type in this position matches the argument type in the function type
    };

    for (int i = 0; i < argTypes.size(); ++i) {
        if (argTypes[i] != funcType->parameters[i]) {
            std::cerr << "Function argument types do not match type definition" << std::endl;
            std::exit(TypeCheckUtils::ErrorCodes::FUNCTION_ERROR);
        }
    }
    // TODO: Check that types align with the argument types specified in the symbol table
    // Finally, set the type of this node to equal the return type of the function
    this->type = funcType->returnType;
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
