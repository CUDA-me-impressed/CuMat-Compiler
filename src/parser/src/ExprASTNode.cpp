#include "ExprASTNode.hpp"

// static llvm::AllocaInst* CreateNamedAlloca(llvm::Function* fn,
//                                           Types::TypeDecl* type,
//                                           const std::string& name) {
//    llvm::BasicBlock* bb = builder.GetInsertBlock();
//    llvm::BasicBlock::iterator ip = builder.GetInsertPoint();
//
//    llvm::IRBuilder<> tmpBuilder(&fn->getEntryBlock(),
//                                 fn->getEntryBlock().begin());
//    llvm::Type* type = ty->LlvmType();
//
//    llvm::AllocaInst* a = tmpBuilder.CreateAlloca(type, 0, name);
//    int align = std::max(ty->AlignSize(), 8);
//    if (a->getAlignment() < align) {
//        a->setAlignment(align);
//    }
//
//    // Now go back to where we were...
//    builder.SetInsertPoint(bb, ip);
//    return a;
//}