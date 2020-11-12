//
// Created by tobyl on 12/11/2020.
//
#include "CuMatVisitor.hpp"


antlrcpp::Any CuMatVisitor::visitProgram(
    CuMatGrammarParser::ProgramContext *ctx) {
    return CuMatGrammarBaseVisitor::visitProgram(ctx);
}
