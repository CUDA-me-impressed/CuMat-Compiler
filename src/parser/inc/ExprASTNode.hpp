#pragma once

#include "ASTNode.hpp"
#include "Type.hpp"

namespace Analysis {
class DimensionSymbolTable;
}

namespace AST {
class ExprNode : public Node {
   public:
    std::shared_ptr<Typing::Type> type;

    [[nodiscard]] std::shared_ptr<Typing::Type> getType() const;
    void setType(std::shared_ptr<Typing::Type> ty);

    [[nodiscard]] std::string toTree(const std::string& prefix, const std::string& childPrefix) const override;

    [[nodiscard]] virtual bool isConst() const noexcept { return false; }
    [[nodiscard]] virtual std::vector<std::shared_ptr<AST::ExprNode>> constData(
        std::shared_ptr<AST::ExprNode>& me) const {
        throw std::runtime_error("attempt to access constData on non-const node");
    };
    [[nodiscard]] virtual bool isLiteralNode() const noexcept { return false; };
};
}  // namespace AST
