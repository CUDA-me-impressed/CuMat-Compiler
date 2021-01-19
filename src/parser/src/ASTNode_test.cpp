//
// Created by matt on 24/11/2020.
//

#include "ASTNode_test.hpp"

using namespace trompeloeil;

TEST_CASE("ASTNode default implementation forwards calls to its children", "[AST::Node]") {
    constexpr size_t NUM_MOCKS = 3;

    // object under test
    AST::Node testNode{};

    // mock objects
    std::vector<std::shared_ptr<AST::Test::NodeMock>> mocks;
    for (size_t i = 0; i < NUM_MOCKS; i++) {
        auto mock = std::make_shared<AST::Test::NodeMock>();
        mocks.push_back(mock);
        testNode.addChild(mock);
    }

    // allow REQUIRE_CALLs to live beyond their loop iteration
    using expectation = std::unique_ptr<trompeloeil::expectation>;
    std::vector<expectation> exps;

    SECTION("semanticPass") {
        for (auto& mock : mocks) {
            exps.push_back(NAMED_REQUIRE_CALL(*mock, semanticPass()));
        }

        testNode.semanticPass();
    }

    SECTION("codeGen") {
        // create a blank context
        Utils::IRContext ctx{};

        for (auto& mock : mocks) {
            exps.push_back(NAMED_REQUIRE_CALL(*mock, codeGen(&ctx)).RETURN(nullptr));
        }

        REQUIRE(testNode.codeGen(&ctx) == nullptr);
    }
}
