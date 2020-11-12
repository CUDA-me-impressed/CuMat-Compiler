//
// Created by tobyl on 12/11/2020.
//
#include "CuMatVisitor.hpp"

#include "ASTNode.hpp"

antlrcpp::Any CuMatVisitor::visitProgram(
    CuMatGrammarParser::ProgramContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitImports(
    CuMatGrammarParser::ImportsContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitCmimport(
    CuMatGrammarParser::CmimportContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitDefinitions(
    CuMatGrammarParser::DefinitionsContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitDefinition(
    CuMatGrammarParser::DefinitionContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitFuncdef(
    CuMatGrammarParser::FuncdefContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitSignature(
    CuMatGrammarParser::SignatureContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitArguments(
    CuMatGrammarParser::ArgumentsContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitArgument(
    CuMatGrammarParser::ArgumentContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitTypespec(
    CuMatGrammarParser::TypespecContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitDimensionspec(
    CuMatGrammarParser::DimensionspecContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitBlock(CuMatGrammarParser::BlockContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitAssignment(
    CuMatGrammarParser::AssignmentContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitVarname(
    CuMatGrammarParser::VarnameContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExpression(
    CuMatGrammarParser::ExpressionContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_logic(
    CuMatGrammarParser::Exp_logicContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_comp(
    CuMatGrammarParser::Exp_compContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_bit(
    CuMatGrammarParser::Exp_bitContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_sum(
    CuMatGrammarParser::Exp_sumContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_mult(
    CuMatGrammarParser::Exp_multContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_pow(
    CuMatGrammarParser::Exp_powContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_mat(
    CuMatGrammarParser::Exp_matContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_neg(
    CuMatGrammarParser::Exp_negContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_not(
    CuMatGrammarParser::Exp_notContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_chain(
    CuMatGrammarParser::Exp_chainContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_func(
    CuMatGrammarParser::Exp_funcContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitArgs(CuMatGrammarParser::ArgsContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitValue(CuMatGrammarParser::ValueContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitMatrixliteral(
    CuMatGrammarParser::MatrixliteralContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitRowliteral(
    CuMatGrammarParser::RowliteralContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitScalarliteral(
    CuMatGrammarParser::ScalarliteralContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitVariable(
    CuMatGrammarParser::VariableContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitCmnamespace(
    CuMatGrammarParser::CmnamespaceContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitCmtypedef(
    CuMatGrammarParser::CmtypedefContext *ctx) {
    auto n = std::make_shared<ASTNode>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<ASTNode>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::defaultResult() { return nullptr; }

antlrcpp::Any CuMatVisitor::aggregateResult(antlrcpp::Any aggregate,
                                            const antlrcpp::Any &nextResult) {
    if (aggregate.isNull()) {
        std::vector<std::shared_ptr<ASTNode>> container;
        return container;
    }

    aggregate.as<std::vector<std::shared_ptr<ASTNode>>>().push_back(
        nextResult.as<std::shared_ptr<ASTNode>>());
    return aggregate;
}
