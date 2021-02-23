#pragma once

#include <catch2/catch.hpp>
#include <catch2/trompeloeil.hpp>

#include "ASTNode.hpp"
#include "BinaryExprNode.hpp"

namespace AST::Test {

class NodeMock : public AST::Node {
   public:
    MAKE_MOCK0(semanticPass, void(), override) voidsemanticPass(Utils::IRContext* context);
    MAKE_MOCK1(codeGen, llvm::Value*(Utils::IRContext*), override);
};

}  // namespace AST::Test
