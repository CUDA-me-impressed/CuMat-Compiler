#pragma once

#include "ASTNode.hpp"
#include "Type.hpp"

namespace Analysis {
class NameTable;
}

namespace AST {
class ExprAST : public Node {
   public:
    std::shared_ptr<Typing::Type> type;

    virtual ~ExprAST() = default;
    void dimensionPass(Analysis::NameTable* nt) override;
};
}  // namespace AST
