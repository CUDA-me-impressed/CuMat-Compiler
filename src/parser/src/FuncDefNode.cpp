#include "FuncDefNode.hpp"
#include "TypeCheckingUtils.hpp"

#include <iostream>
#include <variant>
#include <utility>

#include "DimensionPass.hpp"

#include "DimensionsSymbolTable.hpp"
#include "TreePrint.hpp"

llvm::Value* AST::FuncDefNode::codeGen(Utils::IRContext* context) {
    // Let us generate a new function -> We will first generate the function argument types
    std::vector<llvm::Type*> argTypes;
    std::vector<std::shared_ptr<Typing::Type>> typesRaw;
    for (const auto& typeNamePair : this->parameters) {
        typesRaw.push_back(typeNamePair.second);
        argTypes.push_back(std::get<Typing::MatrixType>(*typeNamePair.second).getLLVMType(context)->getPointerTo());
    }

    context->symbolTable->enterFunction(funcName);

    // Get out the type and create a function
    auto mt = std::get<Typing::MatrixType>(*this->returnType);

    // We get a pointer to the matrix header type
    auto* mtType = mt.getLLVMType(context)->getPointerTo();
    llvm::FunctionType* ft = llvm::FunctionType::get(mtType, argTypes, false);
    llvm::Function* func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, this->funcName, context->module);

    // Add paramaters to the symbol table
    int i = 0;
    for(auto funcitt = func->arg_begin(); funcitt != func->arg_end(); funcitt++, i++){
        context->symbolTable->setValue(this->parameters[i].second, (llvm::Value*) funcitt, this->parameters[i].first, funcName, "");
    }

    context->symbolTable->setFunctionData(funcName, typesRaw, func);
    context->function = func;

    auto* funcRet = this->block->codeGen(context);

    // Pop the function as we leave the definition of the code
    context->symbolTable->escapeFunction();
    Utils::setNVPTXFunctionType(context, this->funcName, Utils::FunctionCUDAType::Host, func);
    return funcRet;
}

void AST::FuncDefNode::semanticPass(Utils::IRContext* context) {

    // TODO: Put the args into symbol table temporarily for the block semantic pass - remove before end of function
    std::vector<std::shared_ptr<Typing::Type>> typesRaw;
    for (const auto& typeNamePair : this->parameters) {
        if ((typeNamePair.second.get())->index() == 0) {
            std::cerr << "Cannot have functions as arguments" << std::endl;
            std::exit(TypeCheckUtils::ErrorCodes::FUNCTION_ERROR);
        }
        context->semanticSymbolTable->storeVarType(typeNamePair.first, typeNamePair.second);
        typesRaw.push_back(typeNamePair.second);
    }

    // Store within the symbol table
    context->symbolTable->addNewFunction(funcName, typesRaw);

    this->block->semanticPass(context);

    auto blockType = std::get_if<Typing::MatrixType>(&*this->block->returnExpr->type);
    auto returnType = std::get_if<Typing::MatrixType>(&*this->returnType);

    if (blockType->getPrimitiveType() != returnType->getPrimitiveType()) {
        std::cerr << "Return type must match declaration (check for implicit upcasting in binary operators)" << std::endl;
        std::exit(TypeCheckUtils::ErrorCodes::FUNCTION_ERROR);
    }

    // Pop the function as we leave the definition of the code
    context->symbolTable->escapeFunction();

    // Check if the function name is already in use
    if (context->semanticSymbolTable->inFuncTable(this->funcName, "")) {
        TypeCheckUtils::alreadyDefinedError(this->funcName, false);
    }
    // Construct the function type and store it
    auto type = TypeCheckUtils::makeFunctionType(this->returnType, typesRaw);
    context->semanticSymbolTable->storeFuncType(this->funcName, "", type);
    for (const auto& typeNamePair : this->parameters) {
        context->semanticSymbolTable->removeVarEntry(typeNamePair.first);
    }
}

std::string AST::FuncDefNode::toTree(const std::string& prefix, const std::string& childPrefix) const {
    using namespace Tree;
    std::string str{prefix + std::string{"Function Definition: "} + funcName + " ("};
    for (auto const& node : this->parameters) {
        str += printType(*std::get<1>(node)) + " " + std::get<0>(node);
        if (&node != &this->parameters.back()) {
            str += ", ";
        }
    }
    str += ")->" + printType(*returnType) + "\n";
    str += block->toTree(childPrefix + L, childPrefix + B);
    return str;
}

void AST::FuncDefNode::dimensionPass(Analysis::DimensionSymbolTable* nt) {
    auto inner_nt = std::move(nt->push_scope());
    for (auto& [name, type] : this->parameters) {
        inner_nt->add_node(name, type);
    }
    this->block->dimensionPass(inner_nt.get());

    auto* rettype = std::get_if<Typing::MatrixType>(this->returnType.get());
    auto* blocktype = std::get_if<Typing::MatrixType>(this->block->returnExpr->type.get());

    if (rettype && blocktype) {
        if (rettype->dimensions == blocktype->dimensions) {
            dimension_error("Return expression doesn't match declared signature", this);
        }
    }
}
void AST::FuncDefNode::dimensionNamePass(Analysis::DimensionSymbolTable* nt) {
    if (auto a = std::get_if<Typing::MatrixType>(this->returnType.get())) {
        if (a->dimensions.empty()) {
            a->dimensions.emplace_back(1);
        }
    }
    nt->add_node(this->funcName, this->returnType);
}
