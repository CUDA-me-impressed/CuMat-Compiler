#include "parser.hpp"

#include <iostream>
#include <strstream>

#include "antlr4-runtime.h"
#include "CuMatGrammarLexer.h"
#include "CuMatGrammarParser.h"

int test_parser(const char* filename) {
    std::ifstream stream;
    stream.open(filename);

    antlr4::ANTLRInputStream input(stream);
    CuMatGrammarLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    CuMatGrammarParser parser(&tokens);

    parser.removeErrorListeners();
    SimpleErrorListener el;
    parser.addErrorListener(&el);

    CuMatGrammarParser::FileContext* tree = parser.file();

    // Fill token buffer
    tokens.fill();

    // Print Tokens
    std::cout << "BEGIN TOKENS\n";
    for (const auto& token : tokens.getTokens()) {
        std::cout << token->toString() << "\n";
    }
    std::cout << "END TOKENS\n" << std::endl;

    try {
        antlr4::tree::ParseTree* tree = parser.program();
        std::cout << tree->toStringTree(true) << std::endl;
        return 0;
    } catch (std::invalid_argument &e) {
        std::cout << e.what() << std::endl;
        return 1;
    }
}

void SimpleErrorListener::syntaxError(antlr4::Recognizer* recognizer,
                                      antlr4::Token* offendingSymbol,
                                      size_t line, size_t charPositionInLine,
                                      const std::string& msg,
                                      std::exception_ptr e) {

    std::ostrstream s;
    s << "At " << line << ":" << charPositionInLine << ", error " << msg;
    throw std::invalid_argument(s.str());
}
