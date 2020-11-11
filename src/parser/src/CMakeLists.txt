list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../antlr4")

if (NOT ANTLR_EXECUTABLE)
    set(ANTLR_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/../antlr4/antlr-4.8-complete.jar")
endif ()

set(ANTLR4_TAG 4.8)
include(ExternalAntlr4Cpp)
find_package(ANTLR REQUIRED)

set(cpp_srcs
        parser.cpp)

set(grammar_srcs
        CuMatGrammar.g4)

# Add ANTLR targets to build the .g4 files
antlr_target(A4Grammar ${grammar_srcs} LEXER PARSER)

# Create a library for the generated sources
add_library(parser_gen STATIC
        ${ANTLR_A4Grammar_CXX_OUTPUTS})
set_property(TARGET parser_gen PROPERTY CXX_STANDARD 11)
target_include_directories(parser_gen PUBLIC
        ${ANTLR_A4Grammar_OUTPUT_DIR})

# link it to the runtime
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
        parser_gen)
target_link_libraries(Parser PUBLIC
        parser_gen)

# create a (temporary) test executable
add_executable(Parser_test
        parser_test.cpp)
target_link_libraries(Parser_test PRIVATE
        Parser)
