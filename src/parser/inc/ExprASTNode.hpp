#pragma once

#include "ASTNode.hpp"
#include "Type.hpp"

namespace AST {
    class ExprAST : public Node {
    public:
        std::shared_ptr<Typing::Type> type;

        virtual ~ExprAST() {}

        virtual void codegen();
    };
}
