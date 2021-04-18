#include "BlockNode.hpp"

#include "TreePrint.hpp"

llvm::Value* AST::BlockNode::codeGen(Utils::IRContext* context) {
    // For this function, we need a new BasicBlock structure
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(context->module->getContext(), this->callingFunctionName + "_entry",
                                                    context->function);
    context->Builder->SetInsertPoint(bb);

    // Loop over each assignment in order
    for (const auto& ass : this->assignments) {
        ass->codeGen(context);
    }

    // Generate Return statement code
    llvm::Value* returnExprVal = this->returnExpr->codeGen(context);

    // Ensure that matrix literals are upcast
    if(auto* rValType = std::get_if<Typing::MatrixType>(&*returnExpr->type)){
        if(rValType->rank == 0){
            returnExprVal = Utils::upcastLiteralToMatrix(context, *rValType, returnExprVal);
        }
    }

    printIfMainFunction(context, returnExprVal);

    llvm::Value* retVal = context->Builder->CreateRet(returnExprVal);

    return retVal;
}

void AST::BlockNode::semanticPass(Utils::IRContext* context) {
    // Run semantic pass on all related nodes - block is not an Expr node, so does not have a type to assign
    for (auto const& assignment : this->assignments) assignment->semanticPass(context);
    this->returnExpr->semanticPass(context);
}
std::string AST::BlockNode::toTree(const std::string& prefix, const std::string& childPrefix) const {
    using namespace Tree;
    std::string str{prefix + std::string{"Block\n"}};
    for (auto const& node : this->assignments) {
        str += node->toTree(childPrefix + T, childPrefix + I);
    }
    str += returnExpr->toTree(childPrefix + L, childPrefix + B);
    return str;
}

/**
 * Handles the printing of main functions on return
 * @param context
 * @param returnExprVal
 */
void AST::BlockNode::printIfMainFunction(Utils::IRContext* context, llvm::Value* returnExprVal) {
    if (this->callingFunctionName == "main") {
        // lmao this is awful code deal with it
        if (auto retType = std::get_if<Typing::MatrixType>(&*this->returnExpr->type)) {
            std::vector<llvm::Value*> argVals({returnExprVal});

            if (retType->primType == Typing::PRIMITIVE::INT) {
                context->Builder->CreateCall(context->symbolTable->printFunctions.funcInt, argVals);
            } else if (retType->primType == Typing::PRIMITIVE::FLOAT) {
                context->Builder->CreateCall(context->symbolTable->printFunctions.funcFloat, argVals);
            } else {
                throw std::runtime_error("Main return type not valid");
            }
        } else if(auto retTypeFunc = std::get_if<Typing::FunctionType>(&*this->returnExpr->type)){
            if(auto retType = std::get_if<Typing::MatrixType>(&*retTypeFunc->returnType)) {
                // Functions still return a matrix

                std::vector<llvm::Value*> argVals({returnExprVal});

                if (retType->primType == Typing::PRIMITIVE::INT) {
                    context->Builder->CreateCall(context->symbolTable->printFunctions.funcInt, argVals);
                } else if (retType->primType == Typing::PRIMITIVE::FLOAT) {
                    context->Builder->CreateCall(context->symbolTable->printFunctions.funcFloat, argVals);
                } else {
                    throw std::runtime_error("Main return type not valid");
                }
            }
        }
    }
}

