#pragma once

#include <memory>

#include "ExprASTNode.hpp"
#include "NameTable.hpp"

namespace Analysis {
class NameTable;
}

namespace AST {
class TernaryExprASTNode : public ExprAST {
    std::shared_ptr<ExprAST> condition, truthy, falsey;

    void codeGen(llvm::Module* module) override;
    void dimensionPass(Analysis::NameTable* nt) override;
};
}  // namespace AST