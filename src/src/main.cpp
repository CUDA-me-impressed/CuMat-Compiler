#include <iostream>
#include <vector>
#include <string>

int main(int argc, char* argv[], char* envp[]) {
    std::vector<std::string> args;
    //First argument is always name of exe, ignore
    for (int i = 1; i < argc; ++i) {
        args.emplace_back(std::string(argv[i]));
    }

    if (2<=>3 == std::strong_ordering::less)
        std::cout << "Hello, World!" << std::endl;

    for (auto arg : args) {
        std::cout << arg << " argument applied" << std::endl;
    }
    return 0;
}
