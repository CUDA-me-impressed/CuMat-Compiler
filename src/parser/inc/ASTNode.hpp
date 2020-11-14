//
// Created by tobyl on 12/11/2020.
//
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace AST {
class Node {
   public:
    std::string literalText;

    std::vector<std::unique_ptr<Node>> children;

    explicit Node(std::string textRep);

    void addChild(std::unique_ptr<Node> n);

    std::string toString() const;

    // Default implementations just call the function on their children
    virtual void semanticPass();
    virtual void codeGen();
};
}  // namespace AST
