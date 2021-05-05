//
// Created by tobyl on 12/11/2020.
//
#include "CuMatVisitor.hpp"

#include <exception>

#include "ASTNode.hpp"
#include "AssignmentNode.hpp"
#include "BinaryExprNode.hpp"
#include "CuMatLexer.h"
#include "CustomTypeDefNode.hpp"
#include "FuncDefNode.hpp"
#include "FunctionExprNode.hpp"
#include "InputFileNode.hpp"
#include "ImportsNode.hpp"
#include "LiteralNode.hpp"
#include "MatrixNode.hpp"
#include "ProgramNode.hpp"
#include "TernaryExprNode.hpp"
#include "UnaryExprNode.hpp"
#include "VariableNode.hpp"

template <class T>
std::shared_ptr<T> pConv(std::shared_ptr<AST::Node> n) {
    return std::static_pointer_cast<T>(n);
}

antlrcpp::Any CuMatVisitor::visitProgram(CuMatParser::ProgramContext* ctx) {
    auto n = std::make_shared<AST::ProgramNode>();
    n->literalText = ctx->getText();

    if (ctx->imports() != nullptr) {
        auto i = visit(ctx->imports());
        n->addChild(std::move(i));
    }

    auto d = visit(ctx->definitions());
    n->addChild(std::move(d));

    return std::move(n);
}

antlrcpp::Any CuMatVisitor::visitImports(CuMatParser::ImportsContext* ctx) {
    auto n = std::make_shared<AST::ImportsNode>();
    n->literalText = ctx->getText();
    for (auto& import : ctx->cmimport()) {
        auto path = import->path()->getText();
        n->importPaths.emplace_back(path);
    }

    return std::move(n);
}

antlrcpp::Any CuMatVisitor::visitDefinitions(CuMatParser::DefinitionsContext* ctx) {
    auto n = std::make_shared<AST::Node>(ctx->getText());
    auto defs = ctx->definition();
    for (auto& def : defs) {
        auto d = visit(def);
        n->addChild(std::move(d));
    }
    return std::move(n);
}

antlrcpp::Any CuMatVisitor::visitDefinition(CuMatParser::DefinitionContext* ctx) {
    if (ctx->funcdef() != nullptr) {
        return std::move(visit(ctx->funcdef()));
    }

    if (ctx->cmtypedef() != nullptr) {
        return std::move(visit(ctx->cmtypedef()));
    }

    if (ctx->assignment() != nullptr) {
        return std::move(visit(ctx->assignment()));
    }

    throw std::runtime_error("No definition found");
}

// TODO Check if anything extra is needed
antlrcpp::Any CuMatVisitor::visitFuncdef(CuMatParser::FuncdefContext* ctx) {
    auto n = std::make_shared<AST::FuncDefNode>();
    n->literalText = ctx->getText();

    auto sig = ctx->signature();

    // Return Type
    n->returnType = std::move(visit(sig->typespec()));

    // FuncName
    n->funcName = ctx->signature()->funcname()->getText();

    // Parameters
    auto paramsCtx = sig->parameters();
    if (paramsCtx) {
        auto paramCtx = paramsCtx->parameter();
        std::vector<std::pair<std::string, std::shared_ptr<Typing::Type>>> paramContainer;
        for (auto& param : paramCtx) {
            std::pair<std::string, std::shared_ptr<Typing::Type>> p(param->varname()->getText(),
                                                                    std::move(visit(param->typespec())));
            paramContainer.emplace_back(p);
        }
        n->parameters = std::vector<std::pair<std::string, std::shared_ptr<Typing::Type>>>(paramContainer);
    }

    // Block
    n->block = std::move(visit(ctx->block()));
    n->block->callingFunctionName = n->funcName;

    return std::move(pConv<AST::Node>(n));
}

// NOTE: Returns a Type instead of a Node
antlrcpp::Any CuMatVisitor::visitTypespec(CuMatParser::TypespecContext* ctx) {
    if (ctx->cmtypename()->primitive() != nullptr) {
        Typing::MatrixType m;
        auto primType = ctx->cmtypename()->primitive();
        if (primType->T_INT()) {
            m.primType = Typing::PRIMITIVE::INT;
        } else if (primType->T_FLOAT()) {
            m.primType = Typing::PRIMITIVE::FLOAT;
        } else if (primType->T_STRING()) {
            m.primType = Typing::PRIMITIVE::STRING;
        } else if (primType->T_BOOL()) {
            m.primType = Typing::PRIMITIVE::BOOL;
        } else if (primType->functype()) {
            Typing::FunctionType f;
            std::vector<std::shared_ptr<Typing::Type>> params;
            for (auto& argspec : primType->functype()->argspecs) {
                params.emplace_back(std::move(visit(argspec)));
            }
            f.parameters = std::vector(params);
            f.returnType = std::move(visit(primType->functype()->retspec));
            return std::make_shared<Typing::Type>(f);
        }
        if (ctx->dimensionspec()) {
            auto specs = ctx->dimensionspec();
            m.rank = specs->dimension().size();
            std::vector<uint> dims;
            for (auto& dim : specs->dimension()) {
                if (dim->INT()) {
                    dims.emplace_back(std::stoi(dim->INT()->getText()));
                } else {
                    // Is this going to be a subtle bug?
                    dims.emplace_back(0);
                }
            }
            m.dimensions = std::vector<uint>(dims);
        } else {
            m.rank = 0;
        }

        return std::make_shared<Typing::Type>(m);
    } else {
        if (ctx->cmtypename()->typeidentifier()->TYPE_ID() != nullptr) {
            // Generic Type
            Typing::GenericType g;
            g.name = ctx->cmtypename()->typeidentifier()->TYPE_ID()->getText();
            return std::make_shared<Typing::Type>(g);
        } else if (ctx->cmtypename()->typeidentifier()->ID() != nullptr) {
            // Custom Type
            Typing::CustomType ct;
            ct.name = ctx->cmtypename()->typeidentifier()->ID()->getText();
            return std::make_shared<Typing::Type>(ct);
        } else {
            throw std::runtime_error("Something in typespec has gone horribly wrong");
        }
    }
}

antlrcpp::Any CuMatVisitor::visitBlock(CuMatParser::BlockContext* ctx) {
    auto n = std::make_shared<AST::BlockNode>();
    n->literalText = ctx->getText();

    std::vector<std::shared_ptr<AST::AssignmentNode>> assigns;
    for (auto& ass : ctx->assignments) {
        assigns.emplace_back(std::move(visit(ass)));
    }
    n->assignments = std::vector<std::shared_ptr<AST::AssignmentNode>>(assigns);

    auto e = visit(ctx->expression());
    n->returnExpr = std::move(e);
    return std::move(n);
}

antlrcpp::Any CuMatVisitor::visitAssignment(CuMatParser::AssignmentContext* ctx) {
    auto n = std::make_shared<AST::AssignmentNode>();
    n->literalText = ctx->getText();

    if (ctx->asstarget()->varname() != nullptr) {
        auto name = ctx->asstarget()->varname()->identifier()->getText();
        n->name = name;
    } else if (ctx->asstarget()->decomp() != nullptr) {
        n->lVal = std::move(visit(ctx->asstarget()->decomp()));
    } else {
        throw std::runtime_error("No Asstarget found at: " + n->literalText);
    }

    n->rVal = std::move(visit(ctx->expression()));

    return std::move(n);
}

antlrcpp::Any CuMatVisitor::visitExpression(CuMatParser::ExpressionContext* ctx) {
    if (ctx->exp_logic() != nullptr) {
        return std::move(visit(ctx->exp_logic()));
    }

    if (ctx->lambda() != nullptr) {
        return std::move(visit(ctx->lambda()));
    }

    if (ctx->exp_if() != nullptr) {
        return std::move(visit(ctx->exp_if()));
    }

    throw std::runtime_error("Expression type not found");
}

antlrcpp::Any CuMatVisitor::visitExp_if(CuMatParser::Exp_ifContext* ctx) {
    auto ifN = std::make_shared<AST::TernaryExprNode>();
    ifN->literalText = ctx->getText();
    ifN->condition = std::move(visit(ctx->expression(0)));
    ifN->truthy = std::move(visit(ctx->expression(1)));
    ifN->falsey = std::move(visit(ctx->expression(2)));

    return std::move(pConv<AST::ExprNode>(ifN));
}

bool CuMatVisitor::compareTokenTypes(size_t a, size_t b) const {
    return this->parserVocab->getSymbolicName(a) == this->parserVocab->getSymbolicName(b);
}

antlrcpp::Any CuMatVisitor::visitExp_logic(CuMatParser::Exp_logicContext* ctx) {
    auto lowerTier = ctx->exp_comp();
    auto ops = ctx->op_logic();
    if (lowerTier.size() > 1) {
        std::shared_ptr<AST::ExprNode> rightSide;
        auto opIt = ops.rbegin();
        for (auto it = lowerTier.rbegin(); it != lowerTier.rend(); it++) {
            auto n = std::make_shared<AST::BinaryExprNode>();
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
                throw std::runtime_error("Encountered unknown operator, or Toby can't code");
            }
            n->lhs = std::move(visit(*it));
            n->literalText = n->lhs->literalText + (op->getText()) + n->rhs->literalText;
            rightSide = std::move(pConv<AST::ExprNode>(n));
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
        std::shared_ptr<AST::ExprNode> rightSide;
        auto opIt = ops.rbegin();
        for (auto it = lowerTier.rbegin(); it != lowerTier.rend(); it++) {
            auto n = std::make_shared<AST::BinaryExprNode>();
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
                throw std::runtime_error("Encountered unknown operator, or Toby can't code");
            }
            n->lhs = std::move(visit(*it));
            n->literalText = n->lhs->literalText + (op->getText()) + n->rhs->literalText;
            rightSide = std::move(pConv<AST::ExprNode>(n));
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
        std::shared_ptr<AST::ExprNode> rightSide;
        auto opIt = ops.rbegin();
        for (auto it = lowerTier.rbegin(); it != lowerTier.rend(); it++) {
            auto n = std::make_shared<AST::BinaryExprNode>();
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
                throw std::runtime_error("Encountered unknown operator, or Toby can't code");
            }
            n->lhs = std::move(visit(*it));
            n->literalText = n->lhs->literalText + (op->getText()) + n->rhs->literalText;
            rightSide = std::move(pConv<AST::ExprNode>(n));
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
        std::shared_ptr<AST::ExprNode> rightSide;
        auto opIt = ops.rbegin();
        for (auto it = lowerTier.rbegin(); it != lowerTier.rend(); it++) {
            auto n = std::make_shared<AST::BinaryExprNode>();
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
                throw std::runtime_error("Encountered unknown operator, or Toby can't code");
            }
            n->lhs = std::move(visit(*it));
            n->literalText = n->lhs->literalText + (op->getText()) + n->rhs->literalText;
            rightSide = std::move(pConv<AST::ExprNode>(n));
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
        std::shared_ptr<AST::ExprNode> rightSide;
        auto opIt = ops.rbegin();
        for (auto it = lowerTier.rbegin(); it != lowerTier.rend(); it++) {
            auto n = std::make_shared<AST::BinaryExprNode>();
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
                throw std::runtime_error("Encountered unknown operator, or Toby can't code");
            }
            n->lhs = std::move(visit(*it));
            n->literalText = n->lhs->literalText + (op->getText()) + n->rhs->literalText;
            rightSide = std::move(pConv<AST::ExprNode>(n));
        }
        return std::move(rightSide);
    } else {
        return std::move(visit(lowerTier.front()));
    }
}

antlrcpp::Any CuMatVisitor::visitExp_pow(CuMatParser::Exp_powContext* ctx) {
    auto lowerTier = ctx->exp_mat();
    if (lowerTier.size() > 1) {
        std::shared_ptr<AST::ExprNode> leftSide;
        for (auto& it : lowerTier) {
            auto n = std::make_shared<AST::BinaryExprNode>();
            if (leftSide == nullptr) {
                leftSide = std::move(visit(it));
                continue;  // Skip the last one so that we can setup the loop
                // properly
            }

            n->lhs = std::move(leftSide);
            n->op = AST::BIN_OPERATORS::POW;
            n->rhs = std::move(visit(it));
            n->literalText =
                n->lhs->literalText + (this->parserVocab->getLiteralName(CuMatLexer::POW)) + n->rhs->literalText;
            leftSide = std::move(pConv<AST::ExprNode>(n));
        }
        return std::move(leftSide);
    } else {
        return std::move(visit(lowerTier.front()));
    }
}

antlrcpp::Any CuMatVisitor::visitExp_mat(CuMatParser::Exp_matContext* ctx) {
    auto lowerTier = ctx->exp_neg();
    if (lowerTier.size() > 1) {
        std::shared_ptr<AST::ExprNode> rightSide;
        for (auto it = lowerTier.rbegin(); it != lowerTier.rend(); it++) {
            auto n = std::make_shared<AST::BinaryExprNode>();
            if (rightSide == nullptr) {
                rightSide = std::move(visit(*it));
                continue;  // Skip the last one so that we can setup the loop
                // properly
            }

            n->rhs = std::move(rightSide);
            n->op = AST::BIN_OPERATORS::MATM;
            n->lhs = std::move(visit(*it));
            n->literalText =
                n->lhs->literalText + (this->parserVocab->getLiteralName(CuMatLexer::MATM)) + n->rhs->literalText;
            rightSide = std::move(pConv<AST::ExprNode>(n));
        }
        return std::move(rightSide);
    } else {
        return std::move(visit(lowerTier.front()));
    }
}

antlrcpp::Any CuMatVisitor::visitExp_neg(CuMatParser::Exp_negContext* ctx) {
    auto negations = ctx->op_neg();
    // Quick optimisation for silly scenarios like -----3 to become -3
    if (negations.size() % 2 == 0) {
        return std::move(visit(ctx->exp_bnot()));
    } else {
        auto n = std::make_shared<AST::UnaryExprNode>();
        n->literalText = ctx->getText();
        n->op = AST::UNA_OPERATORS::NEG;
        n->operand = std::move(visit(ctx->exp_bnot()));
        return std::move(pConv<AST::ExprNode>(n));
    }
}

antlrcpp::Any CuMatVisitor::visitExp_bnot(CuMatParser::Exp_bnotContext* ctx) {
    auto negations = ctx->op_bnot();
    // Quick optimisation for silly scenarios like .!.!.!3 to become .!3
    if (negations.size() % 2 == 0) {
        return std::move(visit(ctx->exp_not()));
    } else {
        auto n = std::make_shared<AST::UnaryExprNode>();
        n->literalText = ctx->getText();
        n->op = AST::UNA_OPERATORS::BNOT;
        n->operand = std::move(visit(ctx->exp_not()));
        return std::move(pConv<AST::ExprNode>(n));
    }
}

antlrcpp::Any CuMatVisitor::visitExp_not(CuMatParser::Exp_notContext* ctx) {
    auto negations = ctx->op_not();
    // Quick optimisation for silly scenarios like !!!!3 to become 3
    if (negations.size() % 2 == 0) {
        return std::move(visit(ctx->exp_chain()));
    } else {
        auto n = std::make_shared<AST::UnaryExprNode>();
        n->literalText = ctx->getText();
        n->op = AST::UNA_OPERATORS::LNOT;
        n->operand = std::move(visit(ctx->exp_chain()));
        return std::move(pConv<AST::ExprNode>(n));
    }
}

antlrcpp::Any CuMatVisitor::visitExp_chain(CuMatParser::Exp_chainContext* ctx) {
    auto lowerTier = ctx->exp_func();
    if (lowerTier.size() > 1) {
        std::shared_ptr<AST::ExprNode> leftSide;
        for (auto& it : lowerTier) {
            auto n = std::make_shared<AST::BinaryExprNode>();
            if (leftSide == nullptr) {
                leftSide = std::move(visit(it));
                continue;  // Skip the last one so that we can setup the loop
                // properly
            }

            n->lhs = std::move(leftSide);
            n->op = AST::BIN_OPERATORS::CHAIN;
            n->rhs = std::move(visit(it));
            n->literalText =
                n->lhs->literalText + (this->parserVocab->getLiteralName(CuMatLexer::CHAIN)) + n->rhs->literalText;
            leftSide = std::move(pConv<AST::ExprNode>(n));
        }
        return std::move(leftSide);
    } else {
        return std::move(visit(lowerTier.front()));
    }
}

antlrcpp::Any CuMatVisitor::visitExp_func(CuMatParser::Exp_funcContext* ctx) {
    if(ctx->value()->variable() != nullptr && (ctx->value()->variable()->varname()->identifier()->ID()->getText() == "readInts" || ctx->value()->variable()->varname()->identifier()->ID()->getText() == "readFloats"))
    {
        auto iN = std::make_shared<AST::InputFileNode>();
        iN->literalText = ctx->getText();
        auto numArgs = ctx->args().size();
        if(numArgs < 2)
        {
            throw std::runtime_error("Not enough arguments to the Input Function");
        }

        auto fileNameArg = ctx->args()[0];

        if(fileNameArg->expression().size() > 1)
        {
            throw std::runtime_error("First argument must not be a tuple");
        }

        auto fileNameNode = visit(fileNameArg->expression(0));
        try {
            std::shared_ptr<AST::ExprNode> eNode = std::move(fileNameNode);
            auto fileNameN = std::move(pConv<AST::LiteralNode<std::string>>(eNode));
            iN->fileName = std::move(fileNameN->value);
        } catch (std::bad_cast b) {
            throw std::runtime_error("First argument must be a string literal");
        }

        Typing::MatrixType t;

        t.rank = ctx->args().size() - 1;

        if(ctx->value()->variable()->varname()->identifier()->ID()->getText() == "readInts")
        {
            t.primType = Typing::PRIMITIVE::INT;
        } else if (ctx->value()->variable()->varname()->identifier()->ID()->getText() == "readFloats")
        {
            t.primType = Typing::PRIMITIVE::FLOAT;
        } else
        {
            throw std::runtime_error("Something went wrong with establishing which of the input functions is in use.");
        }

        //Mostly repeat for each subsequent argument
        for(int i = 1; i < numArgs; ++i)
        {
            auto dimArg = ctx->args()[i];
            if(dimArg->expression().size() > 1)
            {
                throw std::runtime_error("The dimensions must not be tuples");
            }

            auto dimStatement = visit(dimArg->expression(0));
            try {
                std::shared_ptr<AST::ExprNode> eNode = std::move(dimStatement);
                auto dimN = std::move(pConv<AST::LiteralNode<int>>(eNode));
                t.dimensions.push_back(dimN->value);
            } catch (std::runtime_error re) {
                throw std::runtime_error("There was an issue with the: " + std::to_string(i) + " argument to input");
            }
        }

        iN->type = std::make_shared<Typing::Type>(t);

        return std::move(pConv<AST::ExprNode>(iN));
    }

    auto fN = std::make_shared<AST::FunctionExprNode>();
    fN->literalText = ctx->getText();
    // If no arguments applied, just pass it up
    if (ctx->args().empty()) {
        return std::move(visit(ctx->value()));
    }
    auto value = visit(ctx->value());
    fN->nonAppliedFunction = std::move(value);
    if (!ctx->args().empty()) {
        std::vector<std::shared_ptr<AST::ExprNode>> arguments;
        for (auto arg : ctx->args()) {
            for (auto a : arg->expression()) {
                arguments.emplace_back(std::move(visit(a)));
            }
        }
        fN->args = std::move(arguments);  // This...might be an issue and need
        // to use the copy semantics. We'll see
    }

    return std::move(pConv<AST::ExprNode>(fN));
}

antlrcpp::Any CuMatVisitor::visitValue(CuMatParser::ValueContext* ctx) {
    if (ctx->literal() != nullptr) {
        if (ctx->literal()->matrixliteral() != nullptr) {
            return std::move(visit(ctx->literal()->matrixliteral()));
        } else {
            return std::move(visit(ctx->literal()->scalarliteral()));
        }
    } else if (ctx->expression() != nullptr) {
        return std::move(visit(ctx->expression()));
    } else {
        return std::move(visit(ctx->variable()));
    }
}

antlrcpp::Any CuMatVisitor::visitMatrixliteral(CuMatParser::MatrixliteralContext* ctx) {
    auto mN = std::make_shared<AST::MatrixNode>();
    mN->literalText = ctx->getText();
    Typing::MatrixType t;
    std::vector<uint> dimensions;
    std::vector<uint> seps;
    std::vector<std::shared_ptr<AST::ExprNode>> values;
    int inDimension = 0;
    if(ctx->rowliteral() == nullptr) //Empty matrix literal
    {
        t.rank = 0;
        mN->type = std::make_shared<Typing::Type>(t);
        return std::move(pConv<AST::ExprNode>(mN));
    }
    dimensions.push_back(ctx->rowliteral()->cols.size());  // First size
    for (auto exp : ctx->rowliteral()->cols) {
        values.emplace_back(std::move(visit(exp)));
        seps.emplace_back(1);
    }
    seps.pop_back();  // Fix fencepost error
    if (!ctx->dimensionLiteral().empty()) {
        for (auto dim : ctx->dimensionLiteral()) {
            auto dimension = dim->BSLASH().size();
            seps.emplace_back(dimension + 1);
            /*if (dimension > inDimension) {  // Above
                while (dimension >= dimensions.size()) {
                    dimensions.push_back(2);
                    inDimension++;
                }
            } else if (dimension == inDimension) {  // More rows/layers etc.
                dimensions[dimension]++;
            } else if (dimension < inDimension) {  // Check that they match up to earlier
                if (dim->rowliteral()->cols.size() != dimensions[dimension - 1]) {
                    throw std::runtime_error("Dimensions do not match up: Dimension:" + (std::to_string(dimension)) +
                                             " inDimension: " + (std::to_string(inDimension)));
                } else {
                    inDimension = dimension;
                }
            }
            */
            for (auto exp : dim->rowliteral()->cols) {
                values.emplace_back(std::move(visit(exp)));
                seps.emplace_back(1);
            }
            seps.pop_back();  // Fix fencepost error
        }
    }

    mN->data = std::move(values);
    mN->separators = std::move(seps);
    t.dimensions = std::vector<uint>(dimensions);
    t.rank = dimensions.size();
    mN->type = std::make_shared<Typing::Type>(t);
    return std::move(pConv<AST::ExprNode>(mN));
}

antlrcpp::Any CuMatVisitor::visitScalarliteral(CuMatParser::ScalarliteralContext* ctx) {
    if (ctx->stringliteral() != nullptr) {
        auto n = std::make_shared<AST::LiteralNode<std::string>>();
        n->literalText = ctx->getText();
        n->value = ctx->stringliteral()->STRING()->getText();
        Typing::MatrixType mn;
        mn.rank = 0;
        mn.primType = Typing::PRIMITIVE::STRING;
        n->type = std::make_shared<Typing::Type>(mn);
        return std::move(pConv<AST::ExprNode>(n));
    } else  // Implies numLiteral is not a nullptr
    {
        if (ctx->numliteral()->INT() != nullptr) {
            auto n = std::make_shared<AST::LiteralNode<int>>();
            n->literalText = ctx->getText();
            n->value = std::stoi(ctx->numliteral()->INT()->getText());
            Typing::MatrixType mn;
            mn.rank = 0;
            mn.primType = Typing::PRIMITIVE::INT;
            n->type = std::make_shared<Typing::Type>(mn);
            return std::move(pConv<AST::ExprNode>(n));
        } else  // Implies float
        {
            auto n = std::make_shared<AST::LiteralNode<float>>();
            n->literalText = ctx->getText();
            n->value = std::stof(ctx->numliteral()->FLOAT()->getText());
            Typing::MatrixType mn;
            mn.rank = 0;
            mn.primType = Typing::PRIMITIVE::FLOAT;
            n->type = std::make_shared<Typing::Type>(mn);
            return std::move(pConv<AST::ExprNode>(n));
        }
    }
}

antlrcpp::Any CuMatVisitor::visitVariable(CuMatParser::VariableContext* ctx) {
    auto n = std::make_shared<AST::VariableNode>();
    n->literalText = ctx->getText();
    n->name = ctx->varname()->identifier()->getText();

    if (ctx->cmnamespace() != nullptr) {
        for (auto id : ctx->cmnamespace()->identifier()) {
            n->namespacePath.emplace_back(id->ID()->getText());
        }
    }

    if (ctx->slice() != nullptr) {
        n->variableSlicing = std::move(visit(ctx->slice()));
    }

    return std::move(pConv<AST::ExprNode>(n));
}

antlrcpp::Any CuMatVisitor::visitCmtypedef(CuMatParser::CmtypedefContext* ctx) {
    auto n = std::make_shared<AST::CustomTypeDefNode>();
    n->literalText = ctx->getText();

    n->name = ctx->newtype()->identifier()->getText();
    for (auto& attr : ctx->attrblock()->attrs) {
        n->attributes.emplace_back(std::move(visit(attr)));
    }

    return std::move(n);
}

antlrcpp::Any CuMatVisitor::visitAttr(CuMatParser::AttrContext* ctx) {
    auto n = std::make_shared<AST::TypeDefAttributeNode>();
    n->literalText = ctx->getText();

    n->name = ctx->attrname()->identifier()->getText();
    n->attrType = std::move(visitTypespec(ctx->typespec()));

    return std::move(n);
}

antlrcpp::Any CuMatVisitor::visitDecomp(CuMatParser::DecompContext* ctx) {
    auto n = std::make_shared<AST::DecompNode>();
    n->literalText = ctx->getText();
    n->lVal = ctx->varname()->getText();

    if (ctx->asstarget()->varname() != nullptr) {
        n->rVal = std::variant<std::string, std::shared_ptr<AST::DecompNode>>(ctx->asstarget()->varname()->getText());
    } else if (ctx->asstarget()->decomp() != nullptr) {
        n->rVal = std::variant<std::string, std::shared_ptr<AST::DecompNode>>(
            std::move(visitDecomp(ctx->asstarget()->decomp())));
    } else {
        throw std::runtime_error("Something has gone wrong in Decomposition");
    }
    return std::move(n);
}

antlrcpp::Any CuMatVisitor::visitSlice(CuMatParser::SliceContext* ctx) {
    auto n = std::make_shared<AST::SliceNode>();
    n->literalText = ctx->getText();
    for (auto se : ctx->sliceelement()) {
        if (se->STAR() != nullptr) {
            n->slices.emplace_back(std::variant<bool, std::vector<int>>(true));
        } else {
            std::vector<int> slicingNums;
            for (auto i : se->INT()) {
                slicingNums.emplace_back(std::stoi(i->getText()));
            }
            n->slices.emplace_back(std::variant<bool, std::vector<int>>(slicingNums));
        }
    }
    return std::move(n);
}
