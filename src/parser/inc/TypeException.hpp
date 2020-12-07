#pragma once

#include <llvm/IR/Type.h>

#include <iostream>
#include <sstream>

namespace Typing {

void wrongTypeException(std::string message, llvm::Type* expected, llvm::Type* actual);
void mismatchTypeException(std::string message);

}  // namespace Typing
