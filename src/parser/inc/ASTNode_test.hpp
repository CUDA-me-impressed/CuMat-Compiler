#pragma once

#include <catch2/catch.hpp>
#include <catch2/trompeloeil.hpp>

#include "ASTNode.hpp"
#include "BinaryExprNode.hpp"

namespace AST::Test {

class NodeMock : public AST::Node {
   public:
    MAKE_MOCK1(semanticPass, void(Utils::IRContext*), override);
    MAKE_MOCK1(codeGen, llvm::Value*(Utils::IRContext*), override);
};

}  // namespace AST::Test
