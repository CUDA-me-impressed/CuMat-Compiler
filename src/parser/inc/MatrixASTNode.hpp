#pragma once

#include <vector>

#include "ExprASTNode.hpp"
#include "LiteralASTNode.hpp"
#include "Type.hpp"

namespace AST {
class MatrixASTNode : public ExprAST {
   public:
    std::vector<std::vector<std::shared_ptr<ExprAST>>> data;

    int numElements();
    void codeGen(llvm::Module* module) override;
    void dimensionPass() override;

};
}  // namespace AST
