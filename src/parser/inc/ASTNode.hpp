#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <memory>
#include <string>
#include <vector>

#include "CodeGenUtils.hpp"
#include "Type.hpp"
namespace Analysis {
class DimensionSymbolTable;
}

namespace AST {
class Node {
   public:
    std::string literalText;

    std::vector<std::shared_ptr<Node>> children;

    explicit Node(std::string textRep);

    Node() = default;

    void addChild(std::shared_ptr<Node> n);

    [[nodiscard]] std::string toString() const;

    // Default implementations just call the function on their children
    virtual void semanticPass(Utils::IRContext* context);

    virtual llvm::Value* codeGen(Utils::IRContext* context);
    virtual void dimensionPass(Analysis::DimensionSymbolTable* nt);

    [[nodiscard]] virtual std::string toTree(const std::string& prefix, const std::string& childPrefix) const;
};
}  // namespace AST
