#pragma once

#include <memory>

#include "ExprASTNode.hpp"

namespace AST {
class TernaryExprASTNode : public ExprAST {
    std::unique_ptr<ExprAST> condition, truthy, falsey;
};
}  // namespace AST