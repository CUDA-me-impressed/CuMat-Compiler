#include "ProgramNode.hpp"

llvm::Value* AST::ProgramNode::codeGen(Utils::IRContext* context) {
    context->symbolTable->generateCUDAExternFunctions(context);
    return Node::codeGen(context);
}
void AST::ProgramNode::dimensionPass(Analysis::DimensionSymbolTable* nt) {
    for (auto& node : children) {
        // Register names into symbol table
        node->dimensionNamePass(nt);
    }
    for (auto& node : children) {
        node->dimensionPass(nt);
    }
}
