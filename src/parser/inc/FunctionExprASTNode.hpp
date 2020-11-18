#pragma once

#include <string>
#include <vector>

#include "ExprASTNode.hpp"

namespace AST {
class FunctionExprASTNode : public ExprAST {
   public:
    const std::string funcName;
    std::vector<std::shared_ptr<ExprAST>> args;

    void codeGen() override;
};
}  // namespace AST