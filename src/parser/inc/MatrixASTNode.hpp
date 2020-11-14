#pragma once

#include <vector>

#include "ExprASTNode.hpp"
#include "LiteralASTNode.hpp"
#include "Type.hpp"

namespace AST {
class MatrixASTNode : public ExprAST {
   public:
    std::vector<std::vector<ExprAST>> data;
    void codegen();
};
}  // namespace AST
