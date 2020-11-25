list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../antlr4")

if (NOT ANTLR_EXECUTABLE)
    set(ANTLR_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/../antlr4/antlr-4.8-complete.jar")
endif ()

set(ANTLR4_TAG 4.8)
include(ExternalAntlr4Cpp)
find_package(ANTLR REQUIRED)

set(cpp_srcs
        CuMatASTGenerator.cpp ASTNode.cpp CuMatVisitor.cpp BinaryExprASTNode.cpp FunctionExprASTNode.cpp LiteralASTNode.cpp MatrixASTNode.cpp TernaryExprASTNode.cpp UnaryExprASTNode.cpp)

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
target_link_libraries(Parser PUBLIC
        llvm_interface
        parser_gen)
