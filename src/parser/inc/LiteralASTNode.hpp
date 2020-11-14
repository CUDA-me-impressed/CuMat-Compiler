#pragma once

#include "ExprASTNode.hpp"

namespace AST {
    template<class T>
    class LiteralASTNode : public ExprAST {
    public:
        T value;

        void codegen();
    };
}  // namespace AST
