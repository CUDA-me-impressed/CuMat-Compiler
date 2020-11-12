//
// Created by matt on 08/11/2020.
//

#ifndef CUMAT_COMPILER_PARSER_HPP
#define CUMAT_COMPILER_PARSER_HPP

#include <antlr4-runtime.h>
#include "ASTNode.hpp"

class SimpleErrorListener : public antlr4::BaseErrorListener {
    void syntaxError(antlr4::Recognizer *recognizer,
                     antlr4::Token *offendingSymbol, size_t line,
                     size_t charPositionInLine, const std::string &msg,
                     std::exception_ptr e) override;
};

std::shared_ptr<ASTNode> runParser(std::string fileName);

#endif  // CUMAT_COMPILER_PARSER_HPP