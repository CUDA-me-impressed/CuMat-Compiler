add_executable(CuMatComp main.cpp)

target_include_directories(CuMatComp PUBLIC ../inc)
target_link_libraries(CuMatComp PUBLIC
        llvm_interface
        Parser
        PreProcessor)

