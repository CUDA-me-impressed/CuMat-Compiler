//
// Created by matt on 08/11/2020.
//
#pragma once
#include <antlr4-runtime.h>

#include "ASTNode.hpp"

class SimpleErrorListener : public antlr4::BaseErrorListener {
    void syntaxError(antlr4::Recognizer* recognizer,
                     antlr4::Token* offendingSymbol, size_t line,
                     size_t charPositionInLine, const std::string& msg,
                     std::exception_ptr e) override;
};

std::shared_ptr<AST::Node> runParser(const std::string& fileName);
