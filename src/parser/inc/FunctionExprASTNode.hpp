#pragma once

#include <string>
#include <vector>

#include "ExprASTNode.hpp"
#include "NameTable.hpp"

namespace Analysis {
class NameTable;
}

namespace AST {
class FunctionExprASTNode : public ExprAST {
   public:
    const std::string funcName;
    std::vector<std::shared_ptr<ExprAST>> args;

    void codeGen(llvm::Module* module) override;
    void dimensionPass(Analysis::NameTable* nt) override;
};
}  // namespace AST