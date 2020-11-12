//
// Created by tobyl on 12/11/2020.
//

#ifndef _CUMATPARSER_HPP_
#define _CUMATPARSER_HPP_

#include <antlr4-runtime.h>
#include "CuMatGrammarBaseVisitor.h"

class CuMatVisitor : public CuMatGrammarBaseVisitor
{
    antlrcpp::Any visitProgram(CuMatGrammarParser::ProgramContext *ctx) override;

    antlrcpp::Any visitImports(CuMatGrammarParser::ImportsContext *ctx) override;


    antlrcpp::Any visitDefinitions(CuMatGrammarParser::DefinitionsContext *ctx) override;
    antlrcpp::Any visitDefinition(CuMatGrammarParser::DefinitionContext *ctx) override;

    antlrcpp::Any visitFuncdef(CuMatGrammarParser::FuncdefContext *ctx) override;

    antlrcpp::Any visitAssignment(CuMatGrammarParser::AssignmentContext *ctx) override;
    antlrcpp::Any visitVarname(CuMatGrammarParser::VarnameContext *ctx) override;

    antlrcpp::Any visitExpression(CuMatGrammarParser::ExpressionContext *ctx) override;
    antlrcpp::Any visitExp_logic(CuMatGrammarParser::Exp_logicContext *ctx) override;
    antlrcpp::Any visitExp_comp(CuMatGrammarParser::Exp_compContext *ctx) override;
    antlrcpp::Any visitExp_bit(CuMatGrammarParser::Exp_bitContext *ctx) override;
    antlrcpp::Any visitExp_sum(CuMatGrammarParser::Exp_sumContext *ctx) override;
    antlrcpp::Any visitExp_mult(CuMatGrammarParser::Exp_multContext *ctx) override;
    antlrcpp::Any visitExp_pow(CuMatGrammarParser::Exp_powContext *ctx) override;
    antlrcpp::Any visitExp_mat(CuMatGrammarParser::Exp_matContext *ctx) override;
    antlrcpp::Any visitExp_neg(CuMatGrammarParser::Exp_negContext *ctx) override;
    antlrcpp::Any visitExp_not(CuMatGrammarParser::Exp_notContext *ctx) override;
    antlrcpp::Any visitExp_chain(CuMatGrammarParser::Exp_chainContext *ctx) override;
    antlrcpp::Any visitExp_func(CuMatGrammarParser::Exp_funcContext *ctx) override;
};

#endif  //_CUMATPARSER_HPP_
