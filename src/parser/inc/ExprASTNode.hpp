#pragma once

#include "ASTNode.hpp"
#include "Type.hpp"

namespace AST {
class ExprNode : public Node {
   public:
    std::shared_ptr<Typing::Type> type;

    virtual ~ExprNode() = default;
};
}  // namespace AST
