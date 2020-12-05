#include "BlockNode.hpp"

llvm::Value* AST::BlockNode::codeGen(Utils::IRContext* context) {
    return Node::codeGen(context);
}
