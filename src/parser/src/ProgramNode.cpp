#include "ProgramNode.hpp"

llvm::Value* AST::ProgramNode::codeGen(Utils::IRContext* context) { return Node::codeGen(context); }
