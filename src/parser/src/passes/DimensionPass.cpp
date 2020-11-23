//
// Created by thomas on 22/11/2020.
//
#import "ASTNode.hpp"
#import "BinaryExprASTNode.hpp"
#import "FunctionExprASTNode.hpp"
#import "LiteralASTNode.hpp"
#import "MatrixASTNode.hpp"
#import "TernaryExprASTNode.hpp"
#import "UnaryExprASTNode.hpp"

namespace AST {

void Node::dimensionPass() {
    for (auto const& child : this->children) child->dimensionPass();
}

void BinaryExprASTNode::dimensionPass() {
    switch (this->op) {
        case PLUS:
        case MINUS:
        case MUL:
        case DIV:
        case LOR:
        case LAND:
        case LT:
        case GT:
        case LTE:
        case GTE:
        case EQ:
        case NEQ:
        case BAND:
        case BOR:
        case POW:
        case MATM:
            break;
        case CHAIN:

            break;
    }
}
}

namespace ParsePass {

}