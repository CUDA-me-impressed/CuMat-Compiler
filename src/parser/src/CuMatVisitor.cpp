//
// Created by tobyl on 12/11/2020.
//
#include "CuMatVisitor.hpp"

#include <exception>

#include "ASTNode.hpp"
#include "BinaryExprASTNode.hpp"
#include "CuMatLexer.h"
#include "FunctionExprASTNode.hpp"
#include "LiteralASTNode.hpp"
#include "MatrixASTNode.hpp"
#include "TernaryExprASTNode.hpp"
#include "UnaryExprASTNode.hpp"

// TODO Implement
antlrcpp::Any CuMatVisitor::visitProgram(CuMatParser::ProgramContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitImports(CuMatParser::ImportsContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitCmimport(CuMatParser::CmimportContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitDefinitions(
    CuMatParser::DefinitionsContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitDefinition(
    CuMatParser::DefinitionContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitFuncdef(CuMatParser::FuncdefContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitSignature(CuMatParser::SignatureContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitParameters(
    CuMatParser::ParametersContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitTypespec(CuMatParser::TypespecContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitDimensionspec(
    CuMatParser::DimensionspecContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitBlock(CuMatParser::BlockContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitAssignment(
    CuMatParser::AssignmentContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitVarname(CuMatParser::VarnameContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitExpression(
    CuMatParser::ExpressionContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_if(CuMatParser::Exp_ifContext* ctx) {
    auto ifN = std::make_shared<AST::TernaryExprASTNode>();
    ifN->literalText = ctx->getText();
    ifN->condition = std::move(visit(ctx->expression(0)));
    ifN->truthy = std::move(visit(ctx->expression(1)));
    ifN->falsey = std::move(visit(ctx->expression(2)));

    return std::move(ifN);
}

bool CuMatVisitor::compareTokenTypes(size_t a, size_t b) const {
    return this->parserVocab->getSymbolicName(a) ==
           this->parserVocab->getSymbolicName(b);
}

antlrcpp::Any CuMatVisitor::visitExp_logic(CuMatParser::Exp_logicContext* ctx) {
    auto lowerTier = ctx->exp_comp();
    auto ops = ctx->op_logic();
    if (lowerTier.size() > 1) {
        std::shared_ptr<AST::ExprAST> rightSide;
        auto opIt = ops.rbegin();
        for (auto it = lowerTier.rbegin(); it != lowerTier.rend(); it++) {
            auto n = std::make_shared<AST::BinaryExprASTNode>();
            if (rightSide == nullptr) {
                rightSide = std::move(visit(*it));
                continue;  // Skip the last one so that we can setup the loop
                           // properly
            }
            auto op = (*opIt)->op;
            opIt++;
            n->rhs = std::move(rightSide);
            if (compareTokenTypes(op->getType(), CuMatParser::LAND)) {
                n->op = AST::BIN_OPERATORS::LAND;
            } else if (compareTokenTypes(op->getType(), CuMatParser::LOR)) {
                n->op = AST::BIN_OPERATORS::LOR;
            } else {
                throw std::runtime_error(
                    "Encountered unknown operator, or Toby can't code");
            }
            n->lhs = std::move(visit(*it));
            rightSide = std::move(n);
        }
        return std::move(rightSide);
    } else {
        return std::move(visit(lowerTier.front()));
    }
}

antlrcpp::Any CuMatVisitor::visitExp_comp(CuMatParser::Exp_compContext* ctx) {
    auto lowerTier = ctx->exp_bit();
    auto ops = ctx->op_comp();
    if (lowerTier.size() > 1) {
        std::shared_ptr<AST::ExprAST> rightSide;
        auto opIt = ops.rbegin();
        for (auto it = lowerTier.rbegin(); it != lowerTier.rend(); it++) {
            auto n = std::make_shared<AST::BinaryExprASTNode>();
            if (rightSide == nullptr) {
                rightSide = std::move(visit(*it));
                continue;  // Skip the last one so that we can setup the loop
                           // properly
            }
            auto op = (*opIt)->op;
            opIt++;
            n->rhs = std::move(rightSide);
            if (compareTokenTypes(op->getType(), CuMatParser::LT)) {
                n->op = AST::BIN_OPERATORS::LT;
            } else if (compareTokenTypes(op->getType(), CuMatParser::GT)) {
                n->op = AST::BIN_OPERATORS::GT;
            } else if (compareTokenTypes(op->getType(), CuMatParser::LTE)) {
                n->op = AST::BIN_OPERATORS::LTE;
            } else if (compareTokenTypes(op->getType(), CuMatParser::GTE)) {
                n->op = AST::BIN_OPERATORS::GTE;
            } else if (compareTokenTypes(op->getType(), CuMatParser::EQ)) {
                n->op = AST::BIN_OPERATORS::EQ;
            } else if (compareTokenTypes(op->getType(), CuMatParser::NEQ)) {
                n->op = AST::BIN_OPERATORS::NEQ;
            } else {
                throw std::runtime_error(
                    "Encountered unknown operator, or Toby can't code");
            }
            n->lhs = std::move(visit(*it));
            rightSide = std::move(n);
        }
        return std::move(rightSide);
    } else {
        return std::move(visit(lowerTier.front()));
    }
}

antlrcpp::Any CuMatVisitor::visitExp_bit(CuMatParser::Exp_bitContext* ctx) {
    auto lowerTier = ctx->exp_sum();
    auto ops = ctx->op_bit();
    if (lowerTier.size() > 1) {
        std::shared_ptr<AST::ExprAST> rightSide;
        auto opIt = ops.rbegin();
        for (auto it = lowerTier.rbegin(); it != lowerTier.rend(); it++) {
            auto n = std::make_shared<AST::BinaryExprASTNode>();
            if (rightSide == nullptr) {
                rightSide = std::move(visit(*it));
                continue;  // Skip the last one so that we can setup the loop
                // properly
            }
            auto op = (*opIt)->op;
            opIt++;
            n->rhs = std::move(rightSide);
            if (compareTokenTypes(op->getType(), CuMatParser::BAND)) {
                n->op = AST::BIN_OPERATORS::BAND;
            } else if (compareTokenTypes(op->getType(), CuMatParser::BOR)) {
                n->op = AST::BIN_OPERATORS::BOR;
            } else {
                throw std::runtime_error(
                    "Encountered unknown operator, or Toby can't code");
            }
            n->lhs = std::move(visit(*it));
            rightSide = std::move(n);
        }
        return std::move(rightSide);
    } else {
        return std::move(visit(lowerTier.front()));
    }
}

antlrcpp::Any CuMatVisitor::visitExp_sum(CuMatParser::Exp_sumContext* ctx) {
    auto lowerTier = ctx->exp_mult();
    auto ops = ctx->op_sum();
    if (lowerTier.size() > 1) {
        std::shared_ptr<AST::ExprAST> rightSide;
        auto opIt = ops.rbegin();
        for (auto it = lowerTier.rbegin(); it != lowerTier.rend(); it++) {
            auto n = std::make_shared<AST::BinaryExprASTNode>();
            if (rightSide == nullptr) {
                rightSide = std::move(visit(*it));
                continue;  // Skip the last one so that we can setup the loop
                // properly
            }
            auto op = (*opIt)->op;
            opIt++;
            n->rhs = std::move(rightSide);
            if (compareTokenTypes(op->getType(), CuMatParser::PLUS)) {
                n->op = AST::BIN_OPERATORS::PLUS;
            } else if (compareTokenTypes(op->getType(), CuMatParser::MINUS)) {
                n->op = AST::BIN_OPERATORS::MINUS;
            } else {
                throw std::runtime_error(
                    "Encountered unknown operator, or Toby can't code");
            }
            n->lhs = std::move(visit(*it));
            rightSide = std::move(n);
        }
        return std::move(rightSide);
    } else {
        return std::move(visit(lowerTier.front()));
    }
}

antlrcpp::Any CuMatVisitor::visitExp_mult(CuMatParser::Exp_multContext* ctx) {
    auto lowerTier = ctx->exp_pow();
    auto ops = ctx->op_mult();
    if (lowerTier.size() > 1) {
        std::shared_ptr<AST::ExprAST> rightSide;
        auto opIt = ops.rbegin();
        for (auto it = lowerTier.rbegin(); it != lowerTier.rend(); it++) {
            auto n = std::make_shared<AST::BinaryExprASTNode>();
            if (rightSide == nullptr) {
                rightSide = std::move(visit(*it));
                continue;  // Skip the last one so that we can setup the loop
                // properly
            }
            auto op = (*opIt)->op;
            opIt++;
            n->rhs = std::move(rightSide);
            if (compareTokenTypes(op->getType(), CuMatParser::TIMES) ||
                compareTokenTypes(op->getType(), CuMatParser::STAR)) {
                n->op = AST::BIN_OPERATORS::MUL;
            } else if (compareTokenTypes(op->getType(), CuMatParser::DIV)) {
                n->op = AST::BIN_OPERATORS::DIV;
            } else {
                throw std::runtime_error(
                    "Encountered unknown operator, or Toby can't code");
            }
            n->lhs = std::move(visit(*it));
            rightSide = std::move(n);
        }
        return std::move(rightSide);
    } else {
        return std::move(visit(lowerTier.front()));
    }
}

antlrcpp::Any CuMatVisitor::visitExp_pow(CuMatParser::Exp_powContext* ctx) {
    auto lowerTier = ctx->exp_mat();
    if (lowerTier.size() > 1) {
        std::shared_ptr<AST::ExprAST> leftSide;
        for (auto& it : lowerTier) {
            auto n = std::make_shared<AST::BinaryExprASTNode>();
            if (leftSide == nullptr) {
                leftSide = std::move(visit(it));
                continue;  // Skip the last one so that we can setup the loop
                // properly
            }

            n->lhs = std::move(leftSide);
            n->op = AST::BIN_OPERATORS::POW;
            n->rhs = std::move(visit(it));
            leftSide = std::move(n);
        }
        return std::move(leftSide);
    } else {
        return std::move(visit(lowerTier.front()));
    }
}

antlrcpp::Any CuMatVisitor::visitExp_mat(CuMatParser::Exp_matContext* ctx) {
    auto lowerTier = ctx->exp_neg();
    if (lowerTier.size() > 1) {
        std::shared_ptr<AST::ExprAST> rightSide;
        for (auto it = lowerTier.rbegin(); it != lowerTier.rend(); it++) {
            auto n = std::make_shared<AST::BinaryExprASTNode>();
            if (rightSide == nullptr) {
                rightSide = std::move(visit(*it));
                continue;  // Skip the last one so that we can setup the loop
                // properly
            }

            n->rhs = std::move(rightSide);
            n->op = AST::BIN_OPERATORS::MATM;
            n->lhs = std::move(visit(*it));
            rightSide = std::move(n);
        }
        return std::move(rightSide);
    } else {
        return std::move(visit(lowerTier.front()));
    }
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitExp_neg(CuMatParser::Exp_negContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitExp_bnot(CuMatParser::Exp_bnotContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitExp_not(CuMatParser::Exp_notContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitExp_chain(CuMatParser::Exp_chainContext* ctx) {
    auto lowerTier = ctx->exp_func();
    if (lowerTier.size() > 1) {
        std::shared_ptr<AST::ExprAST> leftSide;
        for (auto& it : lowerTier) {
            auto n = std::make_shared<AST::BinaryExprASTNode>();
            if (leftSide == nullptr) {
                leftSide = std::move(visit(it));
                continue;  // Skip the last one so that we can setup the loop
                // properly
            }

            n->lhs = std::move(leftSide);
            n->op = AST::BIN_OPERATORS::CHAIN;
            n->rhs = std::move(visit(it));
            leftSide = std::move(n);
        }
        return std::move(leftSide);
    } else {
        return std::move(visit(lowerTier.front()));
    }
}

antlrcpp::Any CuMatVisitor::visitExp_func(CuMatParser::Exp_funcContext* ctx) {
    auto fN = std::make_shared<AST::FunctionExprASTNode>();
    fN->literalText = ctx->getText();
    fN->nonAppliedFunction = std::move(visit(ctx->value()));
    if (!ctx->args().empty()) {
        std::vector<std::shared_ptr<AST::ExprAST>> arguments;
        for (auto arg : ctx->args()) {
            for (auto a : arg->expression()) {
                arguments.emplace_back(std::move(visit(a)));
            }
        }
        fN->args =
            std::move(arguments);  // This...might be an issue and need
                                   // to use the copy semantics. We'll see
    }

    return std::move(fN);
}

// TODO Implement
antlrcpp::Any CuMatVisitor::visitValue(CuMatParser::ValueContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::visitMatrixliteral(
    CuMatParser::MatrixliteralContext* ctx) {
    auto mN = std::make_shared<AST::MatrixASTNode>();
    mN->literalText = ctx->getText();
    auto t = std::make_shared<Typing::MatrixType>();
    std::vector<uint> dimensions;
    std::vector<std::vector<std::shared_ptr<AST::ExprAST>>> values;
    int inDimension = 0;
    dimensions.push_back(ctx->rowliteral()->cols.size());  // First size
    std::vector<std::shared_ptr<AST::ExprAST>> valueContainer;
    for (auto exp : ctx->rowliteral()->cols) {
        valueContainer.emplace_back(std::move(visit(exp)));
    }
    values.emplace_back(std::move(valueContainer));
    valueContainer.clear();
    if (!ctx->dimensionLiteral().empty()) {
        for (auto dim : ctx->dimensionLiteral()) {
            auto dimension = dim->BSLASH().size();
            if (dimension > inDimension) {  // New dimension
                while (dimension < dimensions.size()) {
                    dimensions.push_back(1);
                    inDimension++;
                }
            } else if (dimension == inDimension) {  // More rows/layers etc.
                dimensions[dimension]++;
            } else if (dimension <
                       inDimension) {  // Check that they match up to earlier
                if (dim->rowliteral()->cols.size() !=
                    dimensions[dimension - 1]) {
                    throw std::runtime_error(
                        "Dimensions do not match up: Dimension:" +
                        (std::to_string(dimension)) +
                        " inDimension: " + (std::to_string(inDimension)));
                }
            }

            for (auto exp : dim->rowliteral()->cols) {
                valueContainer.emplace_back(std::move(visit(exp)));
            }
            values.emplace_back(std::move(valueContainer));
            valueContainer.clear();
        }
    }

    mN->data = std::move(values);
    t->dimensions = dimensions;
    t->rank = dimensions.size();
    mN->type = std::move(t);
    return std::move(mN);
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitScalarliteral(
    CuMatParser::ScalarliteralContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitVariable(CuMatParser::VariableContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitCmnamespace(
    CuMatParser::CmnamespaceContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}
// TODO Implement
antlrcpp::Any CuMatVisitor::visitCmtypedef(CuMatParser::CmtypedefContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto children =
        this->visitChildren(ctx).as<std::vector<std::shared_ptr<AST::Node>>>();
    for (auto& child : children) {
        n->addChild(std::move(child));
    }
    return n;
}

antlrcpp::Any CuMatVisitor::defaultResult() { return nullptr; }

antlrcpp::Any CuMatVisitor::aggregateResult(antlrcpp::Any aggregate,
                                            const antlrcpp::Any& nextResult) {
    if (aggregate.isNull()) {
        std::vector<std::shared_ptr<AST::Node>> container;
        return container;
    }

    aggregate.as<std::vector<std::shared_ptr<AST::Node>>>().push_back(
        nextResult.as<std::shared_ptr<AST::Node>>());
    return aggregate;
}
