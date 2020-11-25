//
// Created by matt on 24/11/2020.
//

#ifndef CUMAT_COMPILER_ASTNODE_TEST_HPP
#define CUMAT_COMPILER_ASTNODE_TEST_HPP

#include <catch2/catch.hpp>
#include <catch2/trompeloeil.hpp>

#include "ASTNode.hpp"
#include "BinaryExprASTNode.hpp"

namespace AST::Test {

class NodeMock : public AST::Node {
   public:
    MAKE_MOCK0(semanticPass, void(), override);
    MAKE_MOCK1(codeGen, void(llvm::Module*), override);
};

}

#endif  // CUMAT_COMPILER_ASTNODE_TEST_HPP
