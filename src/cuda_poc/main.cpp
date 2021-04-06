#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>
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

    builder.CreateRetVoid();

    // verify function
    //    std::string error_msg;
    //    llvm::raw_string_ostream error_os{error_msg};
    //    if (llvm::verifyFunction(*func, &error_os)) {
    //        throw std::runtime_error(error_msg);
    //    }

    return func;
}

// from https://stackoverflow.com/a/21802936
std::vector<unsigned char> readFile(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open() || !file.good()) {
        throw std::runtime_error(strerror(errno));
    }

    file.seekg(0, std::ios::end);
    std::streampos size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (!file.good()) {
        throw std::runtime_error(strerror(errno));
    }

    std::vector<unsigned char> vec(size);
    file.read(reinterpret_cast<char*>(vec.data()), size);

    if (!file.good()) {
        throw std::runtime_error(strerror(errno));
    }

    return vec;
}

llvm::GlobalVariable* genCudaFatbinArray(llvm::LLVMContext& llctx, llvm::IRBuilder<>& builder, llvm::Module& module,
                                         std::vector<unsigned char> cuda_fat_binary,
                                         const char* name = "cuda_fatbin_array") {
    auto* u8_t = llvm::IntegerType::getInt8Ty(llctx);

    // create the global array holding the fat binary
    auto* fatbin_array_t = llvm::ArrayType::get(u8_t, cuda_fat_binary.size());
    std::vector<llvm::Constant*> fatbin_constants;
    fatbin_constants.reserve(cuda_fat_binary.size());
    for (char x : cuda_fat_binary) {
        auto* char_const = llvm::ConstantInt::get(u8_t, (uint64_t)x, false);
        fatbin_constants.push_back(char_const);
    }
    auto* fatbin_array_const = llvm::ConstantArray::get(fatbin_array_t, fatbin_constants);

    module.getOrInsertGlobal(name, fatbin_array_t);
    llvm::GlobalVariable* fatbin_array_global = module.getNamedGlobal(name);
    fatbin_array_global->setConstant(true);
    fatbin_array_global->setInitializer(fatbin_array_const);
    fatbin_array_global->setLinkage(llvm::GlobalVariable::InternalLinkage);

    return fatbin_array_global;
}

llvm::Function* genInitHost(llvm::LLVMContext& llctx, llvm::IRBuilder<>& builder, llvm::Module& module,
                            llvm::GlobalVariable* cuda_fatbin_array, const char* name = "init_host") {
    // basic types
    auto* void_t = llvm::Type::getVoidTy(llctx);

    auto* func_t = llvm::FunctionType::get(void_t, false);
    auto* func = llvm::Function::Create(func_t, llvm::Function::ExternalLinkage, name, module);

    return func;
}

int main(int argc, char** argv) {
    // Init llvm context
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();

    llvm::LLVMContext llctx;
    llvm::IRBuilder<> builder(llctx);
    llvm::Module host_module("test_module_host", llctx);
    llvm::Module device_module("test_module_device", llctx);

    std::string err;

    // setup host target
    std::string host_triple = llvm::sys::getDefaultTargetTriple();
    const llvm::Target* host_target = llvm::TargetRegistry::lookupTarget(host_triple, err);
    if (host_target == nullptr) {
        llvm::errs() << "Failed to get host target: " << err;
        return 1;
    }
    const auto* host_cpu = "generic";
    const auto* host_features = "";
    llvm::TargetOptions host_target_options{};
    llvm::Optional<llvm::Reloc::Model> host_reloc_model{};
    auto* host_target_machine =
        host_target->createTargetMachine(host_triple, host_cpu, host_features, host_target_options, host_reloc_model);
    host_module.setTargetTriple(host_triple);
    host_module.setDataLayout(host_target_machine->createDataLayout());

    // setup device target
    std::string nvidia_triple = "nvptx64-nvidia-cuda";
    const llvm::Target* nvidia_target = llvm::TargetRegistry::lookupTarget(nvidia_triple, err);
    if (nvidia_target == nullptr) {
        llvm::errs() << "Failed to get nvidia target: " << err;
        return 1;
    }
    // all of this is basically just copied from what clang uses when compiling to PTX
    const auto* nvidia_sm = "sm_50";
    const auto* nvidia_cm = "compute_50";
    const auto* nvidia_features = "";
    llvm::TargetOptions nvidia_target_options{};
    llvm::Optional<llvm::Reloc::Model> nvidia_reloc_model{llvm::Reloc::Static};
    auto* nvidia_target_machine = nvidia_target->createTargetMachine(nvidia_triple, nvidia_sm, nvidia_features,
                                                                     nvidia_target_options, nvidia_reloc_model);

    device_module.setTargetTriple(nvidia_triple);
    device_module.setDataLayout(nvidia_target_machine->createDataLayout());

    // generate device add function
    genAddDevice(llctx, builder, device_module);

    // filenames
    const char* device_ir_file = "test_module_device.ir";
    const char* device_s_file = "test_module_device.s";
    const char* device_obj_file = "test_module_device.o";
    const char* device_fatbin_file = "test_module_device.fatbin";

    // save device IR
    std::error_code ec{};
    {
        llvm::raw_fd_ostream device_ir_out(device_ir_file, ec);
        if (ec) {
            llvm::errs() << "error opening file for writing: " << ec.message() << "\n";
            return 1;
        }
        device_module.print(device_ir_out, nullptr);
    }

    // compile device module to NVPTX assembly
    {
        llvm::raw_fd_ostream device_s_out(device_s_file, ec);
        if (ec) {
            llvm::errs() << "error opening device assembly file for writing: " << ec.message() << "\n";
            return 1;
        }

        llvm::legacy::PassManager pass;
        if (nvidia_target_machine->addPassesToEmitFile(pass, device_s_out, nullptr, llvm::CGFT_AssemblyFile)) {
            llvm::errs() << "TargetMachine failure";
            return 1;
        }
        pass.run(device_module);
    }

    // assemble PTX assembly to an object file
    {
        std::ostringstream cmd{};
        cmd << "ptxas "
            << "-m64 -O0 -v --gpu-name " << nvidia_sm << " "
            << "--output-file " << device_obj_file << " " << device_s_file;
        llvm::outs() << cmd.str() << "\n";
        if (std::system(cmd.str().c_str()) != 0) {
            llvm::errs() << "failed to execute ptxas\n";
            return 1;
        }
    }

    {
        // create fatbin file
        std::ostringstream cmd{};
        cmd << "fatbinary -64 "
            << "--create " << device_fatbin_file << " "
            << "--image=profile=" << nvidia_sm << ",file=" << device_obj_file << " "
            << "--image=profile=" << nvidia_cm << ",file=" << device_s_file;
        llvm::outs() << cmd.str() << "\n";
        if (std::system(cmd.str().c_str()) != 0) {
            llvm::errs() << "failed to create fat binary\n";
            return 1;
        }
    }

    // read in the fat binary
    try {
        auto fatbin_contents = readFile(device_fatbin_file);
        genCudaFatbinArray(llctx, builder, host_module, fatbin_contents);
    } catch (std::exception& e) {
        llvm::errs() << "failed to read fatbin contents: " << e.what() << "\n";
        return 1;
    }

    // output host ir
    {
        ec.clear();
        llvm::raw_fd_ostream host_ir_out("test_module_host.ir", ec);
        if (ec) {
            llvm::errs() << "error opening file for writing: " << ec.message() << "\n";
            return 1;
        }
        host_module.print(host_ir_out, nullptr);
    }

    // output host object file
}