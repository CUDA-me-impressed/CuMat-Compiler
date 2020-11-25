#pragma once

#include <vector>

#include "ExprASTNode.hpp"
#include "LiteralASTNode.hpp"
#include "Type.hpp"

namespace AST {
class MatrixNode : public ExprNode {
   public:
    std::vector<std::vector<std::shared_ptr<ExprNode>>> data;

    int numElements();
    void codeGen(llvm::Module* module) override;
};
}  // namespace AST
