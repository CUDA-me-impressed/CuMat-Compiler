#pragma once

#include "ExprASTNode.hpp"


namespace AST {
    enum class TYPE {
        STRING, INT, FLOAT
    };

    template<class T>
    class LiteralASTNode : public ExprAST {
    public:
        T value;

        void codegen();
    };
}  // namespace AST
