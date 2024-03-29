list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../antlr4")

if (NOT ANTLR_EXECUTABLE)
    set(ANTLR_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/../antlr4/antlr-4.8-complete.jar")
endif ()

set(ANTLR4_TAG 4.8)
include(ExternalAntlr4Cpp)
find_package(ANTLR REQUIRED)

set(cpp_srcs
        CuMatASTGenerator.cpp ASTNode.cpp CuMatVisitor.cpp BinaryExprNode.cpp ExprASTNode.cpp FuncDefNode.cpp FunctionExprNode.cpp LiteralNode.cpp MatrixNode.cpp TernaryExprNode.cpp UnaryExprNode.cpp Type.cpp CodeGenUtils.cpp BlockNode.cpp AssignmentNode.cpp VariableNode.cpp SymbolTable.cpp ProgramNode.cpp SliceNode.cpp ImportsNode.cpp TypeDefAttributeNode.cpp CustomTypeDefNode.cpp DecompNode.cpp TypeCheckingUtils.cpp ../inc/TypeCheckingUtils.hpp DimensionPass.cpp TreePrint.cpp DimensionSymbolTable.cpp TypeCheckingSymbolTable.cpp)


set(lexer_srcs
        CuMatLexer.g4)
set(parser_srcs
        CuMatParser.g4)

# Add ANTLR targets to build the .g4 files
antlr_target(A4Lexer ${lexer_srcs} LEXER)
antlr_target(A4Parser ${parser_srcs}
        DEPENDS_ANTLR A4Lexer
        COMPILE_FLAGS -lib ${ANTLR_A4Lexer_OUTPUT_DIR}
        PARSER VISITOR)

# Create a library for the generated sources
add_library(parser_gen STATIC
        ${ANTLR_A4Lexer_CXX_OUTPUTS}
        ${ANTLR_A4Parser_CXX_OUTPUTS})
set_property(TARGET parser_gen PROPERTY CXX_STANDARD 11)
target_include_directories(parser_gen PUBLIC
        ${ANTLR_A4Lexer_OUTPUT_DIR}
        ${ANTLR_A4Parser_OUTPUT_DIR})
target_compile_definitions(parser_gen PUBLIC
        -DANTLR4CPP_STATIC=1)
target_include_directories(parser_gen PUBLIC
        ${ANTLR4_INCLUDE_DIRS})
target_link_libraries(parser_gen PRIVATE
        antlr4_static)

# create a target for the parser cpp code
add_library(Parser
        ${cpp_srcs})
target_include_directories(Parser PUBLIC
        ../inc)
target_include_directories(Parser PUBLIC ../../compiler/inc)

target_link_libraries(Parser PUBLIC
        llvm_interface
        parser_gen)

# Create a test target for parser
set(test_srcs
        test_main.cpp
        ASTNode_test.cpp)
add_executable(Parser_test
        ${test_srcs})
target_link_libraries(Parser_test PRIVATE
        Parser
        Catch2
        Trompeloeil)
add_test(NAME Parser_test
        COMMAND Parser_test)
