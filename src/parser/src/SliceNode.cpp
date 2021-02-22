#include "SliceNode.hpp"

llvm::Value* AST::SliceNode::codeGen(Utils::IRContext* context) {
    // TODO: Do for generalised dimensions
//    std::variant<bool, std::vector<int>> variant = (this->slices.at(0));
//    if(std::get_if<bool>(&variant)){
//        return
//    }
    return Node::codeGen(context);
}