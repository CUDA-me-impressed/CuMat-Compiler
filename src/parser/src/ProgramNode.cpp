#include "ProgramNode.hpp"

llvm::Value* AST::ProgramNode::codeGen(Utils::IRContext* context) {
    context->symbolTable->generateCUDAExternFunctions(context);
    return Node::codeGen(context);
}
