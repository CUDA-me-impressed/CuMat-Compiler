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

    std::shared_ptr<ASTNode> parent;
    std::vector<std::shared_ptr<ASTNode>> children;

    ASTNode(std::shared_ptr<ASTNode> creator, std::string textRep);

    virtual std::shared_ptr<ASTNode> semanticPass();
    virtual void codeGen();
};

#endif  //_ASTNODE_HPP_
