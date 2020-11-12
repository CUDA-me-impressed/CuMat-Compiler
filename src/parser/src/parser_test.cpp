#include "parser.hpp"

#define TEST_FILE "src/parser/samples/test.cumat"

int main(int argc, char** argv) {
    const char* filename = TEST_FILE;
    if (argc > 1) {
        filename = argv[1];
    }

    return test_parser(filename);
}
