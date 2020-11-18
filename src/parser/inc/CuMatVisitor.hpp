//
// Created by tobyl on 12/11/2020.
//

#ifndef _CUMATPARSER_HPP_
#define _CUMATPARSER_HPP_

#include <antlr4-runtime.h>

#include "CuMatParserBaseVisitor.h"

class CuMatVisitor : public CuMatParserBaseVisitor {
   public:
    antlrcpp::Any visitProgram(CuMatParser::ProgramContext* ctx) override;

   private:
    antlrcpp::Any visitImports(CuMatParser::ImportsContext* ctx) override;

    antlrcpp::Any visitDefinitions(
        CuMatParser::DefinitionsContext* ctx) override;
    antlrcpp::Any visitDefinition(CuMatParser::DefinitionContext* ctx) override;

    antlrcpp::Any visitFuncdef(CuMatParser::FuncdefContext* ctx) override;

    antlrcpp::Any visitAssignment(CuMatParser::AssignmentContext* ctx) override;
    antlrcpp::Any visitVarname(CuMatParser::VarnameContext* ctx) override;

    antlrcpp::Any visitExpression(CuMatParser::ExpressionContext* ctx) override;
    antlrcpp::Any visitExp_logic(CuMatParser::Exp_logicContext* ctx) override;
    antlrcpp::Any visitExp_comp(CuMatParser::Exp_compContext* ctx) override;
    antlrcpp::Any visitExp_bit(CuMatParser::Exp_bitContext* ctx) override;
    antlrcpp::Any visitExp_sum(CuMatParser::Exp_sumContext* ctx) override;
    antlrcpp::Any visitExp_mult(CuMatParser::Exp_multContext* ctx) override;
    antlrcpp::Any visitExp_pow(CuMatParser::Exp_powContext* ctx) override;
    antlrcpp::Any visitExp_mat(CuMatParser::Exp_matContext* ctx) override;
    antlrcpp::Any visitExp_neg(CuMatParser::Exp_negContext* ctx) override;
    antlrcpp::Any visitExp_not(CuMatParser::Exp_notContext* ctx) override;
    antlrcpp::Any visitExp_chain(CuMatParser::Exp_chainContext* ctx) override;
    antlrcpp::Any visitExp_func(CuMatParser::Exp_funcContext* ctx) override;
};

#endif  //_CUMATPARSER_HPP_
