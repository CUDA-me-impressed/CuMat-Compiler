#pragma once

#include <vector>

#include "ASTNode.hpp"
#include "ExprASTNode.hpp"
#include "LiteralASTNode.hpp"

namespace AST {
class MatrixASTNode : public Node {
   public:
    TYPE type;  // Namespace pls
    std::vector<std::vector<ExprAST>> data;
    void codegen();
};
}  // namespace AST
