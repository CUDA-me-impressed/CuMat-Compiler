#pragma once

#include <vector>

#include "ExprASTNode.hpp"
#include "FunctionExprNode.hpp"

namespace AST {
class FuncDefNode : public Node {
   public:
    // Function Signature
    std::shared_ptr<Typing::Type> returnType;
    std::string funcName;
    std::vector<std::pair<std::string, std::shared_ptr<Typing::Type>>> parameters;

    std::vector<std::shared_ptr<Node>> assignments;
    std::shared_ptr<ExprNode> returnExpr;

    llvm::Value* codeGen(llvm::Module* TheModule, llvm::IRBuilder<>* Builder, llvm::Function* fp) override;
};
}  // namespace AST