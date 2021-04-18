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

    if (this->callingFunctionName == "main") {
        // lmao this is awful code deal with it
        if (auto retType = std::get_if<Typing::MatrixType>(&*this->returnExpr->type)) {
            auto mainRecord = Utils::getMatrixFromPointer(context, returnExprVal);
            llvm::Value* resLenLLVM =
                llvm::ConstantInt::get(llvm::Type::getInt64Ty(context->module->getContext()), retType->getLength());
            std::vector<llvm::Value*> argVals({mainRecord.dataPtr, resLenLLVM});

            if (retType->primType == Typing::PRIMITIVE::INT) {
                auto callRet = context->Builder->CreateCall(context->symbolTable->printFunctions.funcInt, argVals);
            } else if (retType->primType == Typing::PRIMITIVE::FLOAT) {
                auto callRet = context->Builder->CreateCall(context->symbolTable->printFunctions.funcFloat, argVals);
            } else {
                throw std::runtime_error("Main return type not valid");
            }
        }
    }

    llvm::Value* retVal = context->Builder->CreateRet(returnExprVal);

    return retVal;
}

void AST::BlockNode::semanticPass(Utils::IRContext* context) {
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

void AST::BlockNode::dimensionPass(Analysis::DimensionSymbolTable* nt) {
    for (auto& a : this->assignments) {
        a->dimensionPass(nt);
    }
    returnExpr->dimensionPass(nt);
}
