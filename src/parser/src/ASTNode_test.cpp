//
// Created by matt on 24/11/2020.
//

#include "ASTNode_test.hpp"

SCENARIO("AST Nodes propagate calls to their children", "[AST::Node]") {
    constexpr int NUM_MOCKS = 1;

    GIVEN("An AST::Node with some children") {
        AST::Node root;

        for (int i = 0; i < NUM_MOCKS; i++) {
            root.addChild(
                std::shared_ptr<AST::Node>{new AST::Test::NodeMock()});
        }

        WHEN("semanticPass() is called on the root") {
            root.semanticPass();

            THEN("semanticPass() is called once on each child") {
                for (auto& node : root.children) {
                    auto& mock = dynamic_cast<AST::Test::NodeMock&>(*node);
                    REQUIRE_CALL(mock, semanticPass());
                }
            }
        }

        WHEN("codeGen(nullptr) is called on the root") {
            llvm::Module* const ptr = nullptr;
            root.codeGen(ptr);

            THEN("codeGen(nullptr) is called once on each child") {
                for (auto& node : root.children) {
                    auto& mock = dynamic_cast<AST::Test::NodeMock&>(*node);
                    REQUIRE_CALL(mock,
                                 codeGen(trompeloeil::eq<llvm::Module*>(ptr)));
                }
            }
        }
    }
}
