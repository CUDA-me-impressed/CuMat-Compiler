#include "CuMatParser.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <strstream>

#include "CuMatGrammarLexer.h"
#include "CuMatGrammarParser.h"
#include "CuMatVisitor.hpp"
#include "antlr4-runtime.h"

void SimpleErrorListener::syntaxError(antlr4::Recognizer* recognizer,
                                      antlr4::Token* offendingSymbol,
                                      size_t line, size_t charPositionInLine,
                                      const std::string& msg,
                                      std::exception_ptr e) {
    std::ostrstream s;
    s << "At " << line << ":" << charPositionInLine << ", error " << msg;
    throw std::invalid_argument(s.str());
}
std::shared_ptr<ASTNode> runParser(std::string fileName) {
    std::ifstream stream;
    stream.open(fileName);

    antlr4::ANTLRInputStream input(stream);
    CuMatGrammarLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    CuMatGrammarParser parser(&tokens);

    parser.removeErrorListeners();
    SimpleErrorListener el;
    parser.addErrorListener(&el);
    CuMatVisitor visitor;

    try {
        CuMatGrammarParser::ProgramContext* context = parser.program();
        auto tree = visitor.visitProgram(context);
        return std::move(tree.as<std::shared_ptr<ASTNode>>());
    } catch (std::invalid_argument& e) {
        std::cout << e.what() << std::endl;
        return nullptr;
    }
}
