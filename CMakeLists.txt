cmake_minimum_required(VERSION 3.12)
project(CuMat_Compiler)

find_program(CLANG_10 "clang++-10")
IF (CLANG_10)
    set(CMAKE_CXX_COMPILER clang++-10)
    set(CMAKE_CXX_CLANG_TID clang-tidy-10)
ELSE ()
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_CXX_CLANG_TIDY clang-tidy)
ENDIF ()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif ()

find_package(LLVM REQUIRED CONFIG
        # try the llvm-10 directory first to avoid other versions
        PATHS /usr/lib/llvm-10/lib/cmake/llvm/)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
# include all of the components, may need to include nvvm at some point
llvm_map_components_to_libnames(llvm_libs support core irreader)
add_library(llvm_interface INTERFACE)
target_link_libraries(llvm_interface INTERFACE ${llvm_libs})
target_include_directories(llvm_interface INTERFACE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(llvm_interface INTERFACE ${LLVM_DEFINITIONS})

link_libraries(-lstdc++fs)
add_subdirectory(src)