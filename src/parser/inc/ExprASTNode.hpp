#pragma once

#include "ASTNode.hpp"
#include "Type.hpp"

namespace Analysis {
class NameTable;
}

namespace AST {
class ExprNode : public Node {
   public:
    std::shared_ptr<Typing::Type> type;

    virtual ~ExprNode() = default;
    void dimensionPass(Analysis::NameTable* nt) override;
};
}  // namespace AST
