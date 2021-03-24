#pragma once

#include "ASTNode.hpp"
#include "Type.hpp"

namespace AST {
class ExprNode : public Node {
   public:
    std::shared_ptr<Typing::Type> type;

    [[nodiscard]] std::shared_ptr<Typing::Type> getType() const;
    void setType(std::shared_ptr<Typing::Type> ty);

    virtual ~ExprNode() = default;
};
}  // namespace AST
