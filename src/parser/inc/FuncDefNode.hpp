#pragma once

#include <vector>

#include "ASTNode.hpp"
#include "BlockNode.hpp"
#include "ExprASTNode.hpp"

namespace AST {
class FuncDefNode : public Node {
   public:
    // Function Signature
    std::shared_ptr<Typing::Type> returnType;
    std::string funcName;
    std::vector<std::pair<std::string, std::shared_ptr<Typing::Type>>> parameters;

    std::shared_ptr<BlockNode> block;

    llvm::Value* codeGen(llvm::Module* TheModule, llvm::IRBuilder<>* Builder, llvm::Function* fp) override;
};
}  // namespace AST