//
// Created by tobyl on 12/11/2020.
//

#ifndef _CUMATPARSER_HPP_
#define _CUMATPARSER_HPP_

#include <antlr4-runtime.h>

#include "CuMatGrammarBaseVisitor.h"

class CuMatVisitor : public CuMatGrammarBaseVisitor {
   public:
    antlrcpp::Any visitProgram(
        CuMatGrammarParser::ProgramContext *ctx) override;

    antlrcpp::Any visitImports(
        CuMatGrammarParser::ImportsContext *ctx) override;
    antlrcpp::Any visitCmimport(
        CuMatGrammarParser::CmimportContext *ctx) override;

    antlrcpp::Any visitDefinitions(
        CuMatGrammarParser::DefinitionsContext *ctx) override;
    antlrcpp::Any visitDefinition(
        CuMatGrammarParser::DefinitionContext *ctx) override;

    antlrcpp::Any visitFuncdef(
        CuMatGrammarParser::FuncdefContext *ctx) override;
    antlrcpp::Any visitSignature(
        CuMatGrammarParser::SignatureContext *ctx) override;
    antlrcpp::Any visitArguments(
        CuMatGrammarParser::ArgumentsContext *ctx) override;
    antlrcpp::Any visitArgument(
        CuMatGrammarParser::ArgumentContext *ctx) override;
    antlrcpp::Any visitTypespec(
        CuMatGrammarParser::TypespecContext *ctx) override;
    antlrcpp::Any visitDimensionspec(
        CuMatGrammarParser::DimensionspecContext *ctx) override;

    antlrcpp::Any visitBlock(CuMatGrammarParser::BlockContext *ctx) override;

    antlrcpp::Any visitAssignment(
        CuMatGrammarParser::AssignmentContext *ctx) override;
    antlrcpp::Any visitVarname(
        CuMatGrammarParser::VarnameContext *ctx) override;

    antlrcpp::Any visitExpression(
        CuMatGrammarParser::ExpressionContext *ctx) override;
    antlrcpp::Any visitExp_logic(
        CuMatGrammarParser::Exp_logicContext *ctx) override;
    antlrcpp::Any visitExp_comp(
        CuMatGrammarParser::Exp_compContext *ctx) override;
    antlrcpp::Any visitExp_bit(
        CuMatGrammarParser::Exp_bitContext *ctx) override;
    antlrcpp::Any visitExp_sum(
        CuMatGrammarParser::Exp_sumContext *ctx) override;
    antlrcpp::Any visitExp_mult(
        CuMatGrammarParser::Exp_multContext *ctx) override;
    antlrcpp::Any visitExp_pow(
        CuMatGrammarParser::Exp_powContext *ctx) override;
    antlrcpp::Any visitExp_mat(
        CuMatGrammarParser::Exp_matContext *ctx) override;
    antlrcpp::Any visitExp_neg(
        CuMatGrammarParser::Exp_negContext *ctx) override;
    antlrcpp::Any visitExp_not(
        CuMatGrammarParser::Exp_notContext *ctx) override;
    antlrcpp::Any visitExp_chain(
        CuMatGrammarParser::Exp_chainContext *ctx) override;
    antlrcpp::Any visitExp_func(
        CuMatGrammarParser::Exp_funcContext *ctx) override;

    antlrcpp::Any visitArgs(CuMatGrammarParser::ArgsContext *ctx) override;

    antlrcpp::Any visitValue(CuMatGrammarParser::ValueContext *ctx) override;
    antlrcpp::Any visitMatrixliteral(
        CuMatGrammarParser::MatrixliteralContext *ctx) override;
    antlrcpp::Any visitRowliteral(
        CuMatGrammarParser::RowliteralContext *ctx) override;
    antlrcpp::Any visitScalarliteral(
        CuMatGrammarParser::ScalarliteralContext *ctx) override;

    antlrcpp::Any visitVariable(
        CuMatGrammarParser::VariableContext *ctx) override;
    antlrcpp::Any visitCmnamespace(
        CuMatGrammarParser::CmnamespaceContext *ctx) override;

    antlrcpp::Any visitCmtypedef(
        CuMatGrammarParser::CmtypedefContext *ctx) override;

   protected:
    antlrcpp::Any defaultResult() override;
    // Aggregate results use vectors of type:
    // std::vector<std::shared_ptr<ASTNode>>
    antlrcpp::Any aggregateResult(antlrcpp::Any aggregate,
                                  const antlrcpp::Any &nextResult) override;
};

#endif  //_CUMATPARSER_HPP_
