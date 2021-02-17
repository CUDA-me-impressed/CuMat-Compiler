#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <memory>
#include <vector>

llvm::Function* genAddDevice(llvm::LLVMContext& llctx, llvm::IRBuilder<>& builder, llvm::Module& module,
                             const char* name = "add_device") {
    auto* float_ptr_t = llvm::Type::getFloatPtrTy(llctx);
    auto* float_t = llvm::Type::getFloatTy(llctx);
    auto* void_t = llvm::Type::getVoidTy(llctx);

    // three arguments:
    //   - x (input)
    //   - y (input)
    //   - z (output)
    std::vector<llvm::Type*> argument_types(3, float_ptr_t);

    auto* func_t = llvm::FunctionType::get(void_t, argument_types, false);

    auto* func = llvm::Function::Create(func_t, llvm::Function::ExternalLinkage, name, module);

    // annotate as a kernel
    llvm::SmallVector<llvm::Metadata*, 3> meta{
        llvm::ValueAsMetadata::getConstant(func), llvm::MDString::get(llctx, "kernel"),
        llvm::ValueAsMetadata::getConstant(llvm::ConstantInt::get(llvm::Type::getInt64Ty(llctx), 1))};
    auto* meta_node = llvm::MDTuple::get(llctx, meta);
    module.getOrInsertNamedMetadata("nvvm.annotations")->addOperand(meta_node);

    // set the names to make it easier to read
    std::vector<llvm::Argument*> args;
    for (auto& arg : func->args()) {
        args.push_back(&arg);
    }

    auto* x_v = args[0];
    auto* y_v = args[1];
    auto* z_v = args[2];

    x_v->setName("x");
    y_v->setName("y");
    z_v->setName("z");

    // codegen
    auto* entry_bb = llvm::BasicBlock::Create(llctx, "entry", func);
    builder.SetInsertPoint(entry_bb);

    // get ptx intrinsics for the ptx module
    auto* intr_blockIdx_x = llvm::Intrinsic::getDeclaration(&module, llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
    auto* intr_blockDim_x = llvm::Intrinsic::getDeclaration(&module, llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x);
    auto* intr_threadIdx_x = llvm::Intrinsic::getDeclaration(&module, llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x);

    // call the instrinsics to get the location of the invocation
    auto* blockIdx_x_v = builder.CreateCall(intr_blockIdx_x->getFunctionType(), intr_blockIdx_x);
    auto* blockDim_x_v = builder.CreateCall(intr_blockDim_x->getFunctionType(), intr_blockDim_x);
    auto* threadIdx_x = builder.CreateCall(intr_threadIdx_x->getFunctionType(), intr_threadIdx_x);

    // flatten to a linear index
    auto* index_v = builder.CreateAdd(builder.CreateMul(blockIdx_x_v, blockDim_x_v), threadIdx_x);

    // index the arrays
    auto* x_elem_ptr_v = builder.CreateInBoundsGEP(float_t, x_v, index_v);
    auto* y_elem_ptr_v = builder.CreateInBoundsGEP(float_t, y_v, index_v);
    auto* z_elem_ptr_v = builder.CreateInBoundsGEP(float_t, z_v, index_v);

    // load inputs
    auto* x_elem_v = builder.CreateLoad(x_elem_ptr_v, "xi");
    auto* y_elem_v = builder.CreateLoad(y_elem_ptr_v, "yi");
    // compute output
    auto* z_elem_v = builder.CreateFAdd(x_elem_v, y_elem_v, "zi");
    // store output
    builder.CreateStore(z_elem_v, z_elem_ptr_v);

    // verify function
    //    std::string error_msg;
    //    llvm::raw_string_ostream error_os{error_msg};
    //    if (llvm::verifyFunction(*func, &error_os)) {
    //        throw std::runtime_error(error_msg);
    //    }

    return func;
}

int main(int argc, char** argv) {
    // Init llvm context
    llvm::LLVMContext llctx;
    llvm::IRBuilder<> builder(llctx);
    llvm::Module host_module("test_module_host", llctx);
    llvm::Module device_module("test_module_device", llctx);

    // setup targets
    std::string target_device = "nvptx64-nvidia-cuda";
    std::string target_host = llvm::sys::getDefaultTargetTriple();
    device_module.setTargetTriple(target_device);
    host_module.setTargetTriple(target_host);

    // generate device add function
    genAddDevice(llctx, builder, device_module);

    // output ir
    device_module.print(llvm::outs(), nullptr);
}