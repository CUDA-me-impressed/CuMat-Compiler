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
    antlrcpp::Any visitFuncdef(CuMatGrammarParser::FuncdefContext *ctx) override;
};

#endif  //_CUMATPARSER_HPP_
