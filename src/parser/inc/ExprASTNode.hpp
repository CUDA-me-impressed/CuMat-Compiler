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

    [[nodiscard]] std::shared_ptr<Typing::Type> getType() const;
    void setType(std::shared_ptr<Typing::Type> ty);

    virtual ~ExprNode() = default;
    void dimensionPass(Analysis::NameTable* nt) override;

    [[nodiscard]] std::string toTree(const std::string& prefix, const std::string& childPrefix) const override;
};
}  // namespace AST
