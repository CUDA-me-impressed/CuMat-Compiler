#include "CuMatASTGenerator.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <strstream>

#include "CuMatLexer.h"
#include "CuMatParser.h"
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
std::shared_ptr<AST::Node> runParser(const std::string& fileName) {
    std::ifstream stream;
    stream.open(fileName);

    antlr4::ANTLRInputStream input(stream);
    CuMatLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    CuMatParser parser(&tokens);
    auto vocab = parser.getVocabulary();

    parser.removeErrorListeners();
    SimpleErrorListener el;
    parser.addErrorListener(&el);
    CuMatVisitor visitor;
    visitor.parserVocab = &vocab;

    try {
        CuMatParser::ProgramContext* context = parser.program();
        auto tree = visitor.visitProgram(context);
        return std::move(tree.as<std::shared_ptr<AST::Node>>());
    } catch (std::invalid_argument& e) {
        std::cout << e.what() << std::endl;
        return nullptr;
    }
}
