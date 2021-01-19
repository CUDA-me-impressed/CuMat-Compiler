//
// Created by tobyl on 12/11/2020.
//
#pragma once

#include <antlr4-runtime.h>

#include "CuMatParserBaseVisitor.h"

class CuMatVisitor : public CuMatParserBaseVisitor {
   public:
    antlr4::dfa::Vocabulary* parserVocab;

    antlrcpp::Any visitProgram(CuMatParser::ProgramContext* ctx) override;

    antlrcpp::Any visitImports(CuMatParser::ImportsContext* ctx) override;

    antlrcpp::Any visitCmimport(CuMatParser::CmimportContext* ctx) override;

    antlrcpp::Any visitDefinitions(CuMatParser::DefinitionsContext* ctx) override;

    antlrcpp::Any visitDefinition(CuMatParser::DefinitionContext* ctx) override;

    antlrcpp::Any visitFuncdef(CuMatParser::FuncdefContext* ctx) override;

    antlrcpp::Any visitTypespec(CuMatParser::TypespecContext* ctx) override;

    antlrcpp::Any visitBlock(CuMatParser::BlockContext* ctx) override;

    antlrcpp::Any visitAssignment(CuMatParser::AssignmentContext* ctx) override;

    antlrcpp::Any visitDecomp(CuMatParser::DecompContext* ctx) override;

    antlrcpp::Any visitSlice(CuMatParser::SliceContext* ctx) override;

    antlrcpp::Any visitExpression(CuMatParser::ExpressionContext* ctx) override;

    antlrcpp::Any visitExp_if(CuMatParser::Exp_ifContext* ctx) override;

    antlrcpp::Any visitExp_logic(CuMatParser::Exp_logicContext* ctx) override;

    antlrcpp::Any visitExp_comp(CuMatParser::Exp_compContext* ctx) override;

    antlrcpp::Any visitExp_bit(CuMatParser::Exp_bitContext* ctx) override;

    antlrcpp::Any visitExp_sum(CuMatParser::Exp_sumContext* ctx) override;

    antlrcpp::Any visitExp_mult(CuMatParser::Exp_multContext* ctx) override;

    antlrcpp::Any visitExp_pow(CuMatParser::Exp_powContext* ctx) override;

    antlrcpp::Any visitExp_mat(CuMatParser::Exp_matContext* ctx) override;

    antlrcpp::Any visitExp_neg(CuMatParser::Exp_negContext* ctx) override;

    antlrcpp::Any visitExp_bnot(CuMatParser::Exp_bnotContext* ctx) override;

    antlrcpp::Any visitExp_not(CuMatParser::Exp_notContext* ctx) override;

    antlrcpp::Any visitExp_chain(CuMatParser::Exp_chainContext* ctx) override;

    antlrcpp::Any visitExp_func(CuMatParser::Exp_funcContext* ctx) override;

    antlrcpp::Any visitValue(CuMatParser::ValueContext* ctx) override;

    antlrcpp::Any visitMatrixliteral(CuMatParser::MatrixliteralContext* ctx) override;

    antlrcpp::Any visitScalarliteral(CuMatParser::ScalarliteralContext* ctx) override;

    antlrcpp::Any visitVariable(CuMatParser::VariableContext* ctx) override;

    antlrcpp::Any visitCmtypedef(CuMatParser::CmtypedefContext* ctx) override;

   private:
    [[nodiscard]] bool compareTokenTypes(size_t a, size_t b) const;
};
