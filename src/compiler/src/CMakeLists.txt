add_executable(CuMatComp main.cpp)
target_include_directories(CuMatComp PUBLIC ../inc)

# add LLVM to the project
message(STATUS "Searching for LLVM")
find_package(LLVM REQUIRED CONFIG)      # find from system lib
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# include all of the components, may need to include nvvm at some point
llvm_map_components_to_libnames(llvm_libs support core irreader)

target_link_libraries(CuMatComp PUBLIC PreProcessor Parser ${llvm_libs})