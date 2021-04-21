#pragma once

#include <string>
#include <vector>

#include "ASTNode.hpp"
#include "Type.hpp"

namespace AST {
class DecompNode : public Node {
   public:
    std::string lVal;
    std::variant<std::string, std::shared_ptr<DecompNode>> rVal;
    llvm::Value* codeGen(Utils::IRContext* context) override;

    void semanticPass(Utils::IRContext* context, Typing::PRIMITIVE primType);
    void dimensionPass(Analysis::DimensionSymbolTable* nt, Typing::MatrixType& type);
};
}  // namespace AST
