#pragma once

#include "ASTNode.hpp"
#include "Type.hpp"

namespace AST {
class ExprAST : public Node {
   public:
    std::unique_ptr<Typing::Type> type;

    virtual ~ExprAST() = default;
};
}  // namespace AST
