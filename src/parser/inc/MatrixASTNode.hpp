#pragma once

#include <vector>

#include "Type.hpp"
#include "ASTNode.hpp"
#include "ExprASTNode.hpp"
#include "LiteralASTNode.hpp"

namespace AST {
class MatrixASTNode : public Node {
   public:
    std::unique_ptr<Typing::Type> type;  // Namespace pls
    std::vector<std::vector<ExprAST>> data;
    void codegen();
};
}  // namespace AST
