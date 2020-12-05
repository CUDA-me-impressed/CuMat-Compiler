#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

#include <algorithm>
#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "CompilerOptions.hpp"
#include "CuMatASTGenerator.hpp"
#include "Preprocessor.hpp"

void printArgumentError(std::string message, std::string arg) {
    const std::string helpText =
        "CuMat Compiler Syntax: CuMat inputFile [ -Wall | -Winfo | -Wnone ] [ "
        "-Oall | -Onone | -Oexp ] [ -o outputfile ]";
    std::cout << helpText << std::endl;
    std::cout << message << arg << std::endl;
}

int main(int argc, char* argv[], char* envp[]) {
    std::vector<std::string> args;
    std::string inputFileName;
    std::string outputFile;

    const std::set<std::string> validFlags = {"-Wall", "-Winfo", "-Wnone", "-Oall", "-Onone", "-Oexp", "-o"};

    // First argument is always name of exe, ignore
    for (int i = 1; i < argc; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        auto arg = std::string(argv[i]);

        if (validFlags.find(arg) != validFlags.end()) {
            args.emplace_back(arg);
        } else if (!args.empty() && args.back() == "-o" && outputFile.empty())  // Output and at least one additional
        {
            outputFile = arg;
        } else if (inputFileName.empty()) {
            inputFileName = arg;
        } else {
            printArgumentError("Unrecognised argument", arg);
            return 1;  // Quit early
        }
    }

    if (inputFileName.empty()) {
        printArgumentError("Needs an input file, e.g. ", "main.cm");
        return 1;  // Quit early
    }

    CompilerOptions co;
    if (outputFile.empty()) {
        co = CompilerOptions(inputFileName);
    } else {
        co = CompilerOptions(inputFileName, outputFile);
    }

    // Assign Warning level
    std::vector<std::string> warningLevels;
    std::copy_if(args.begin(), args.end(), std::back_inserter(warningLevels),
                 [](std::string s) { return s.rfind("-W", 0) == 0; });
    if (warningLevels.size() > 1) {
        printArgumentError("Maximum of one warning level to be set", "");
        return 1;
    }
    if (warningLevels.size() == 1) {
        auto warn = warningLevels.front();
        if (warn == "-Wall") {
            co.warningVerbosity = WARNINGS::ALL;
        } else if (warn == "-Winfo") {
            co.warningVerbosity = WARNINGS::INFO;
        } else if (warn == "-Wnone") {
            co.warningVerbosity = WARNINGS::NONE;
        } else {  // This shouldn't happen, but for completeness
            printArgumentError("Unrecognised warning level", warn);
            return 1;  // Quit early
        }
    }

    // Assign Optimisation level
    std::vector<std::string> optimiseLevels;
    std::copy_if(args.begin(), args.end(), std::back_inserter(optimiseLevels),
                 [](std::string s) { return s.rfind("-O", 0) == 0; });
    if (optimiseLevels.size() > 1) {
        printArgumentError("Maximum of one optimisation level to be set", "");
        return 1;
    }
    if (optimiseLevels.size() == 1) {
        auto opt = optimiseLevels.front();
        if (opt == "-Oall") {
            co.optimisationLevel = OPTIMISATION::ALL;
        } else if (opt == "-Onone") {
            co.optimisationLevel = OPTIMISATION::NONE;
        } else if (opt == "-Oexp") {
            co.optimisationLevel = OPTIMISATION::EXPERIMENTAL;
        } else {  // This shouldn't happen, but for completeness
            printArgumentError("Unrecognised optimisation level", opt);
            return 1;  // Quit early
        }
    }

    std::cout << "Input file: " << co.inputFile << std::endl;
    std::cout << "Output file: " << co.outputFile << std::endl;
    for (auto arg : args) {
        std::cout << arg << " argument applied" << std::endl;
    }

    Preprocessor::SourceFileLoader loader(inputFileName);
    auto files = loader.load();
    std::vector<std::string> firstCU = files.at(0);
    std::vector<std::tuple<std::string, std::shared_ptr<AST::Node>>> parseTrees;
    for (const auto& file : firstCU) {
        auto tree = runParser(file);
        std::cout << "Parsed: " << file << std::endl;
        parseTrees.emplace_back(std::make_tuple<const std::string, std::shared_ptr<AST::Node>>(
            (const std::basic_string<char, std::char_traits<char>, std::allocator<char>>&&)file, std::move(tree)));
    }

    // Pretty printing for test
    for (const auto& tree : parseTrees) {
        std::cout << "Program Tree: " << std::get<0>(tree) << std::endl;
        std::cout << std::get<1>(tree)->literalText << std::endl;
    }

    llvm::LLVMContext TheContext;
    for (const auto& tree : parseTrees) {
        llvm::Module TheModule("CuMat-" + std::get<0>(tree), TheContext);
        llvm::IRBuilder<> Builder(TheContext);
        // Context containing the module and IR Builder
        Utils::IRContext treeContext = {&TheModule, &Builder};
        std::get<1>(tree)->codeGen(&treeContext);

        std::error_code EC;
        llvm::raw_fd_ostream dest("CuMat-" + std::get<0>(tree) + ".ll", EC);
        std::cout << "Printing" << std::endl;
        treeContext.module->print(dest, nullptr);
    }

    return 0;
}
