#pragma once

#include <string>
#include <vector>

#include "ExprASTNode.hpp"

namespace AST {
class FunctionExprASTNode : public ExprAST {
   public:
    std::shared_ptr<ExprAST> nonAppliedFunction;
    std::vector<std::shared_ptr<ExprAST>> args;

    void codeGen(llvm::Module* module) override;
};
}  // namespace AST