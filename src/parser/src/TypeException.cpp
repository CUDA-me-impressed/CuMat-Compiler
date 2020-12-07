#include "TypeException.hpp"

#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

#include <iostream>
#include <sstream>

void Typing::wrongTypeException (std::string message, llvm::Type* expected, llvm::Type* actual) {

    llvm::raw_ostream &typeOutput = llvm::errs();
    std::ostringstream output;
    output << "Error - " << message << "\nExpected:\n";
    std::cerr << output.str();
    expected->print(typeOutput);
    std::cerr << "\nFound:\n";
    actual->print(typeOutput);

};

void Typing::mismatchTypeException(std::string message) {
    std::ostringstream output;
    output << "Error - " << message << "\n";
    std::cerr << output.str();
};
