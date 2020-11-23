//
// Created by tobyl on 12/11/2020.
//
#pragma once

#include <llvm-10/llvm/ADT/StringRef.h>
#include <llvm-10/llvm/IR/IRBuilder.h>
#include <llvm-10/llvm/IR/LLVMContext.h>
#include <llvm-10/llvm/IR/Module.h>

#include <memory>
#include <string>
#include <vector>

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
    virtual void semanticPass();
    virtual void codeGen(llvm::Module* TheModule, llvm::IRBuilder<>* Builder,
                         llvm::Function* fp);
};
}  // namespace AST
