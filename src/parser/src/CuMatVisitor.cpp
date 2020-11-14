//
// Created by tobyl on 12/11/2020.
//
#include "CuMatVisitor.hpp"

#include "ASTNode.hpp"

antlrcpp::Any CuMatVisitor::visitProgram(CuMatParser::ProgramContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitImports(CuMatParser::ImportsContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitCmimport(CuMatParser::CmimportContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitDefinitions(
    CuMatParser::DefinitionsContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitDefinition(
    CuMatParser::DefinitionContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitFuncdef(CuMatParser::FuncdefContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitSignature(CuMatParser::SignatureContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitArguments(CuMatParser::ArgumentsContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitArgument(CuMatParser::ArgumentContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitTypespec(CuMatParser::TypespecContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitDimensionspec(
    CuMatParser::DimensionspecContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitBlock(CuMatParser::BlockContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitAssignment(
    CuMatParser::AssignmentContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitVarname(CuMatParser::VarnameContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExpression(
    CuMatParser::ExpressionContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_logic(CuMatParser::Exp_logicContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_comp(CuMatParser::Exp_compContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_bit(CuMatParser::Exp_bitContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_sum(CuMatParser::Exp_sumContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_mult(CuMatParser::Exp_multContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_pow(CuMatParser::Exp_powContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_mat(CuMatParser::Exp_matContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_neg(CuMatParser::Exp_negContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_not(CuMatParser::Exp_notContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_chain(CuMatParser::Exp_chainContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_func(CuMatParser::Exp_funcContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitArgs(CuMatParser::ArgsContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitValue(CuMatParser::ValueContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitMatrixliteral(
    CuMatParser::MatrixliteralContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitRowliteral(
    CuMatParser::RowliteralContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitScalarliteral(
    CuMatParser::ScalarliteralContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitVariable(CuMatParser::VariableContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitCmnamespace(
    CuMatParser::CmnamespaceContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitCmtypedef(CuMatParser::CmtypedefContext *ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::defaultResult() { return nullptr; }

antlrcpp::Any CuMatVisitor::aggregateResult(antlrcpp::Any aggregate,
                                            const antlrcpp::Any &nextResult) {
    if (aggregate.isNull()) {
        std::vector<std::shared_ptr<AST::Node>> container;
        return container;
    }

    aggregate.as<std::vector<std::shared_ptr<AST::Node>>>().push_back(
        nextResult.as<std::shared_ptr<AST::Node>>());
    return aggregate;
}
