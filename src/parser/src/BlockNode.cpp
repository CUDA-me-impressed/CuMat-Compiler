#include "BlockNode.hpp"

#include "TreePrint.hpp"

llvm::Value* AST::BlockNode::codeGen(Utils::IRContext* context) {
    // For this function, we need a new BasicBlock structure
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(context->module->getContext(), this->callingFunctionName + "_entry",
                                                    context->function);
    context->Builder->SetInsertPoint(bb);

    // Loop over each assignment in order
    for (const auto& ass : this->assignments) {
        ass->codeGen(context);
    }

    // Generate Return statement code
    llvm::Value* returnExprVal = this->returnExpr->codeGen(context);
    llvm::Value* retVal = context->Builder->CreateRet(returnExprVal);

    return retVal;
}

void AST::BlockNode::semanticPass(Utils::IRContext* context) {
    for (auto const& assignment : this->assignments) assignment->semanticPass(context);
    this->returnExpr->semanticPass(context);
}
std::string AST::BlockNode::toTree(const std::string& prefix, const std::string& childPrefix) const {
    using namespace Tree;
    std::string str{prefix + std::string{"Block\n"}};
    for (auto const& node : this->assignments) {
        str += node->toTree(childPrefix + T, childPrefix + I);
    }
    str += returnExpr->toTree(childPrefix + L, childPrefix + B);
    return str;
}
