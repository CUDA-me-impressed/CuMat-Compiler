#pragma once

#include <vector>

#include "ASTNode.hpp"
#include "ExprASTNode.hpp"
#include "LiteralASTNode.hpp"


class MatrixASTNode : public ASTNode {
public:
    TYPE type;      // Namespace pls
    std::vector<std::vector<ExprAST>> data;
    void codegen();
};