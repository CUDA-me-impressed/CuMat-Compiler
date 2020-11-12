//
// Created by tobyl on 12/11/2020.
//

#ifndef _ASTNODE_HPP_
#define _ASTNODE_HPP_

#include <memory>
#include <string>
#include <vector>

class ASTNode {
   public:
    std::string literalText;

    ASTNode* parent;
    std::vector<std::shared_ptr<ASTNode>> children;

    explicit ASTNode(std::string textRep);

    void addChild(std::shared_ptr<ASTNode> n);

    std::string toString();

    virtual void semanticPass() {}
    virtual void codeGen() {}
};

#endif  //_ASTNODE_HPP_
