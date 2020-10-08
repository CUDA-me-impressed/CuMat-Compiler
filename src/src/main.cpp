#include <iostream>
#include <vector>
#include <string>

int main(int argc, char* argv[], char* envp[]) {
    //First argument is always name of exe, ignore
	std::vector<std::string> args(argv+1, argv+argc);

    //if (2<=>3 == std::strong_ordering::less)
    //    std::cout << "Hello, World!" << std::endl;

	for (auto arg : args) {
        std::cout << arg << " argument applied" << std::endl;
    }
    return 0;
}
