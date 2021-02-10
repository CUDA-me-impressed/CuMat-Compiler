#include "SliceNode.hpp"

llvm::Value* AST::SliceNode::codeGen(Utils::IRContext* context) {

    return Node::codeGen(context);
}