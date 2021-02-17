add_executable(cuda_poc
        main.cpp)
target_link_libraries(cuda_poc PRIVATE llvm_interface Parser)